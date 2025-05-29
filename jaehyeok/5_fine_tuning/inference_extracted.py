import os
import json
import argparse
import logging
import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel
from datasets import Dataset
from datetime import datetime
from fine_tuning_new import load_data, prepare_dataset, split_dataset
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import default_data_collator
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen3-8B LoRA model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--max_input_length", type=int, default=30000,
                        help="Max input length for tokenization")
    parser.add_argument("--max_output_length", type=int, default=1024,
                        help="Max summary length for generation")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Number of beams for generation")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save metrics and outputs")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed evaluation")
    return parser.parse_args()

def custom_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    len_input_ids = torch.stack([item["len_input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    paper_ids = [item["paper_id"] for item in batch]  # 문자열 리스트 유지

    return {
        "input_ids": input_ids,
        "len_input_ids": len_input_ids,
        "labels": labels,
        "paper_id": paper_ids
    }

def compute_and_log_rouge(data, rank, output_dir) -> None:
    log_results = {}
    for item in tqdm(data, desc="Calculating Rouge scores"):
        pid = item["paper_id"]
        gold_summary = item["gold_summary"]
        generated_summary = item["generated_summary"]
        
        rouge_score = {}
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(gold_summary, generated_summary)
        
        rouge_score["rouge1"] = scores["rouge1"].fmeasure
        rouge_score["rouge2"] = scores["rouge2"].fmeasure
        rouge_score["rougeL"] = scores["rougeL"].fmeasure

        log_results[pid] = {
            "gold_summary": gold_summary[:100],
            "generated_summary": generated_summary[:100],
            "rouge_score": rouge_score
        }

    # 평균 계산 및 로그
    avg_rouge1 = sum([result["rouge_score"]["rouge1"] for result in log_results.values()]) / len(log_results)
    avg_rouge2 = sum([result["rouge_score"]["rouge2"] for result in log_results.values()]) / len(log_results)
    avg_rougeL = sum([result["rouge_score"]["rougeL"] for result in log_results.values()]) / len(log_results)
    logging.info(f"Average Rouge scores: rouge1: {avg_rouge1}, rouge2: {avg_rouge2}, rougeL: {avg_rougeL}")
    
    # Save results to JSON file
    rouge_results = {
        "avg_rouge1": avg_rouge1,
        "avg_rouge2": avg_rouge2,
        "avg_rougeL": avg_rougeL,
        "detailed_results": log_results
    }
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fine_tuning_generated_outputs_rank{rank}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(rouge_results, f, indent=2)

    logging.info(f"Saved metrics and predictions to {output_dir}")

def main():
    args = parse_args()

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", args.local_rank)

    logging.basicConfig(level=logging.INFO if args.local_rank == 0 else logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device} (rank {args.local_rank})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    # eos_token_id 안전하게 설정
    if tokenizer.eos_token_id is None:
        eos_token = "</s>"
        logging.info(f"Setting eos_token_id to {eos_token}")
        if eos_token in tokenizer.get_vocab():
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        else:
            raise ValueError("eos_token_id is not set in tokenizer, and default token not found in vocab.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    try:
        model = PeftModel.from_pretrained(model, args.model_dir)
    except Exception:
        pass
    model.to(device)
    model.eval()

    paper_texts, gold_summaries = load_data()
    full_dataset = prepare_dataset(paper_texts, gold_summaries)
    _, test_dataset = split_dataset(full_dataset)

    class SummaryDataset(TorchDataset):
        def __init__(self, input_dataset):
            self.data = input_dataset

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            ex = self.data[idx]
            prompt = f"Summarize the following scientific paper:\n\n{ex['text']}\n\nSummary:"
            input_ids = tokenizer(prompt, max_length=args.max_input_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)
            label_ids = tokenizer(ex["summary"], max_length=args.max_output_length, truncation=True, padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)
            nonpad_len = (input_ids != tokenizer.pad_token_id).sum().item()
            
            return {
                "input_ids": input_ids,
                "len_input_ids": torch.tensor(nonpad_len),
                "labels": label_ids,
                "paper_id": ex["paper_id"]
            }

    dataset = SummaryDataset(test_dataset)
    
    # Check dataset by examining 3 random samples
    if args.local_rank == 0:
        indices = torch.randperm(len(dataset))[:3].tolist()
        logger.info(f"Checking {len(indices)} random samples from dataset:")
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            logger.info(f"Sample {i+1} (paper_id: {sample['paper_id']}):")
            logger.info(f"  Input tokens: {sample['input_ids'].shape}")
            logger.info(f"  Label tokens: {sample['labels'].shape}")
            logger.info(f"  Input text preview: {tokenizer.decode(sample['input_ids'][:50])}...")
            logger.info(f"  Label text preview: {tokenizer.decode(sample['labels'][:50])}...")
            logger.info("-" * 50)
            
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, 
                batch_size=args.batch_size, 
                sampler=sampler, 
                collate_fn=custom_collate_fn)

    all_preds, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, disable=args.local_rank != 0)):
            if i>=1:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_output_length,
                num_beams=args.num_beams,
                no_repeat_ngram_size=2,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_out = outputs.cpu()  # (batch, L)
            prompt_lens = batch["len_input_ids"].cpu()  # (batch, L)
            trimmed = [seq[p_len:].tolist() for seq, p_len in zip(gen_out, prompt_lens)]
            all_preds.extend(trimmed)
            # Store the generated summaries and labels
            if args.local_rank == 0:
                logger.info(f"Generated {len(outputs)} summaries for batch {i+1}")
            all_preds.extend(outputs.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
            all_ids.extend(batch["paper_id"])
    
    # Process all_preds to extract only the generated part (without input prompt)
    local_preds  = tokenizer.batch_decode(all_preds,  skip_special_tokens=True)
    local_labels = tokenizer.batch_decode(all_labels, skip_special_tokens=True)
    local_ids    = all_ids

    # Rank별 결과 저장 (JSON Lines 형식 추천)
    rank_data = [
        {"paper_id": pid, "generated_summary": pred, "gold_summary": gold}
        for pid, pred, gold in zip(local_ids, local_preds, local_labels)
    ]

    # Save the generated summaries to a JSON file
    logging.info(f"Generated summaries for {len(rank_data)} samples in rank {args.local_rank}.")
    compute_and_log_rouge(rank_data, args.local_rank, args.output_dir)

if __name__ == "__main__":
    main()