import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
)
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm

# ---------- utility functions -------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Qwen3-8B LoRA model on a single GPU"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="JaehyeokLee/qwen3-8b-longsumm-extractive-datasets-final",
        help="Path to the fine-tuned model directory (weights + LoRA adapter)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (per device)",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=30_000,
        help="Maximum prompt length (tokens)",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=1_024,
        help="Maximum length of generated summary (new tokens)",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam size for generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save Rouge metrics and generated summaries",
    )
    return parser.parse_args()


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate fn that keeps paper_id as list[str] while stacking tensors.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    len_input_ids = torch.stack([item["len_input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    paper_ids = [item["paper_id"] for item in batch]

    return {
        "input_ids": input_ids,
        "len_input_ids": len_input_ids,
        "labels": labels,
        "paper_id": paper_ids,
    }


def compute_and_log_rouge(
    data: List[Dict[str, str]], output_dir: str, logger: logging.Logger
) -> None:
    """
    Compute ROUGE-1/2/L (F1) for every sample + averages, then dump JSON.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    detailed_results = {}

    for item in tqdm(data, desc="Calculating ROUGE"):
        pid = item["paper_id"]
        scores = scorer.score(item["gold_summary"], item["generated_summary"])
        detailed_results[pid] = {
            "gold_summary": item["gold_summary"][:100],
            "generated_summary": item["generated_summary"][:100],
            "rouge": {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            },
        }

    # averages
    avg = {
        k: sum(v["rouge"][k] for v in detailed_results.values()) / len(detailed_results)
        for k in ["rouge1", "rouge2", "rougeL"]
    }
    logger.info(
        f"Average ROUGE - R1: {avg['rouge1']:.4f}, "
        f"R2: {avg['rouge2']:.4f}, "
        f"RL: {avg['rougeL']:.4f}"
    )

    # save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"eval_{timestamp}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"average": avg, "details": detailed_results}, f, indent=2, ensure_ascii=False
        )
    logger.info(f"Saved metrics and predictions → {out_path}")


# ---------- main --------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------------- #
    # set-up logging & device
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on: {device}")

    # ------------------------------------------------------------------------- #
    # tokenizer & model (base weights + LoRA adapter)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # ensure eos_token_id is set
    if tokenizer.eos_token_id is None:
        eos_token = "</s>"
        if eos_token in tokenizer.get_vocab():
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            logger.info("eos_token_id was missing → set to vocab['</s>']")
        else:
            raise ValueError("Tokenizer lacks eos_token_id and '</s>' token.")

    # load base LM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    # merge / attach LoRA adapter if present
    try:
        model = PeftModel.from_pretrained(model, args.model_dir)
        logger.info("Loaded LoRA adapter successfully.")
    except Exception:
        logger.info("No LoRA adapter found - using base model only.")
    model.to(device).eval()

    # ------------------------------------------------------------------------- #
    # dataset
    from fine_tuning_new import load_data, prepare_dataset, split_dataset  # local import

    paper_texts, gold_summaries = load_data()
    full_ds = prepare_dataset(paper_texts, gold_summaries)
    _, test_ds = split_dataset(full_ds)

    class SummaryDataset(TorchDataset):
        def __init__(self, hf_ds):
            self.data = hf_ds

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            ex = self.data[idx]
            prompt = (
                "Summarize the following scientific paper:\n\n"
                f"{ex['text']}\n\nSummary:"
            )
            input_ids = tokenizer(
                prompt,
                max_length=args.max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            label_ids = tokenizer(
                ex["summary"],
                max_length=args.max_output_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            return {
                "input_ids": input_ids,
                "len_input_ids": torch.tensor((input_ids != tokenizer.pad_token_id).sum().item()),  # for getting pure generations
                "labels": label_ids,
                "paper_id": ex["paper_id"],
            }

    dataset = SummaryDataset(test_ds)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    # inference loop
    all_results = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Generating")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_output_length,
                num_beams=args.num_beams,
                no_repeat_ngram_size=2,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
            )

            # cut off the prompt so only new tokens remain
            prompt_lens = batch["len_input_ids"]
            pure_generations = [g[p_len:].tolist() for g, p_len in zip(generated.cpu(), prompt_lens)]

            preds = tokenizer.batch_decode(pure_generations, skip_special_tokens=True)
            labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            for pid, pred, gold in zip(batch["paper_id"], preds, labels):
                all_results.append(
                    {
                        "paper_id": pid,
                        "generated_summary": pred.strip(),
                        "gold_summary": gold.strip(),
                    }
                )

    logger.info(f"Finished generation for {len(all_results)} samples.")
    compute_and_log_rouge(all_results, args.output_dir, logger)

if __name__ == "__main__":
    main()