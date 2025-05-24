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
from fine_tuning import load_data, prepare_dataset, split_dataset
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
    parser.add_argument("--max_output_length", type=int, default=4096,
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
    labels = torch.stack([item["labels"] for item in batch])
    paper_ids = [item["paper_id"] for item in batch]  # 문자열 리스트 유지

    return {
        "input_ids": input_ids,
        "labels": labels,
        "paper_id": paper_ids
    }

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
            return {
                "input_ids": input_ids,
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
        for batch in tqdm(dataloader, disable=args.local_rank != 0):
            input_ids = batch["input_ids"].to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_output_length,
                num_beams=args.num_beams,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

            all_preds.extend(outputs.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
            all_ids.extend(batch["paper_id"])

    decoded_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(all_labels, skip_special_tokens=True)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(label, pred) for label, pred in zip(decoded_labels, decoded_preds)]

    def avg(metric):
        return sum(score[metric].fmeasure for score in scores) / len(scores)

    metrics = {metric: avg(metric) * 100 for metric in ['rouge1', 'rouge2', 'rougeL']}

    if args.local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "fine_tuning_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        outputs_dict = {pid: pred for pid, pred in zip(all_ids, decoded_preds)}
        with open(os.path.join(args.output_dir, "fine_tuning_generated_outputs.json"), "w") as f:
            json.dump(outputs_dict, f, indent=2)

        logger.info(f"Saved metrics and predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
