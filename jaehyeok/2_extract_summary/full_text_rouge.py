from rouge_score import rouge_scorer
from collections import defaultdict
import json, sys, os, logging, argparse
import random
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

# Create logs directory if it doesn't exist
os.makedirs("./jaehyeok/logs", exist_ok=True)
os.makedirs("./jaehyeok/models", exist_ok=True)

# Configure logging with timestamp in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./jaehyeok/logs/full_text_rouge_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def load_data() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load paper full texts and gold summaries with memory efficiency."""
    logging.info("Loading data...")
    
    # Load paper file paths rather than full contents
    path = "./papers/postprocessed/full_texts/"
    paper_file_paths = {}
    files = os.listdir(path)
    logging.info(f"Found {len(files)} paper files")
    
    for file in files:
        file_id = file.split(".")[0]
        paper_file_paths[file_id] = os.path.join(path, file)
            
    # Load gold summaries
    logging.info("Loading gold summaries")
    gold_summaries = json.load(open("./abstractive_summaries/id_summary_map.json", "r"))
    logging.info(f"Found {len(gold_summaries)} gold summaries")
    
    return paper_file_paths, gold_summaries

def prepare_dataset(paper_file_paths: Dict[str, str], gold_summaries: Dict[str, str]) -> List[Dict[str, str]]:
    """Prepare dataset for fine-tuning with filtering conditions."""
    logging.info("Preparing dataset...")
    
    # Load valid IDs if available
    valid_ids = []
    if os.path.exists("./valid_ids.json"):
        with open("./valid_ids.json", 'r') as f:
            valid_ids = json.load(f)
        logging.info(f"Loaded {len(valid_ids)} valid paper IDs for filtering")
    
    # Track filtering statistics
    filtering_stats = {
        "total_papers": len(paper_file_paths),
        "no_gold_summary": 0,
        "not_in_valid_ids": 0,
        "accepted": 0
    }
    
    # Match papers with their summaries with filtering
    data = []
    for paper_id, paper_path in tqdm(paper_file_paths.items(), desc="Filtering papers"):
        # Read the paper text from file
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logging.warning(f"Error reading paper {paper_id}: {e}")
            continue
        # Skip if not in valid IDs list (if available and not empty)
        if valid_ids and paper_id not in valid_ids:
            # logging.info(f"Skipping {paper_id} - not in valid IDs")
            filtering_stats["not_in_valid_ids"] += 1
            continue
        
        # Skip if no gold summary available
        gold_summary = gold_summaries.get(paper_id, None)
        if gold_summary is None:
            logging.info(f"Skipping {paper_id} - no gold summary found")
            filtering_stats["no_gold_summary"] += 1
            continue
        
        # If we get here, the paper passed all filters
        filtering_stats["accepted"] += 1
        data.append({
            "paper_id": paper_id,
            "text": paper_text,
            "summary": gold_summary
        })
    
    # Log filtering statistics
    logging.info("Filtering statistics:")
    for key, value in filtering_stats.items():
        logging.info(f"  {key}: {value}")
    
    logging.info(f"Created dataset with {len(data)} paper-summary pairs after filtering")
    return data

def compute_and_log_rouge(data) -> None:
    log_results = {}
    for item in tqdm(data, desc="Calculating Rouge scores"):
        pid = item["paper_id"]
        full_text = item["text"]
        gold_summary = item["summary"]

        rouge_score = {}
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(gold_summary, full_text)
        
        rouge_score["rouge1"] = scores["rouge1"].fmeasure
        rouge_score["rouge2"] = scores["rouge2"].fmeasure
        rouge_score["rougeL"] = scores["rougeL"].fmeasure

        log_results[pid] = {
            "gold_summary": gold_summary[:100],
            "full_text": full_text[:100],
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
    with open(f"./jaehyeok/results/rouge_scores_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(rouge_results, f, indent=2)
    # logging.info(f"Rouge scores exported to {results_path}")


def main():
    paper_file_paths, gold_summaries = load_data()
    data = prepare_dataset(paper_file_paths, gold_summaries)

    # Rouge 점수 계산 및 로깅
    logging.info("전체 데이터에 대해 Rouge 점수 계산을 시작합니다.")
    compute_and_log_rouge(data)


if __name__ == "__main__":
    main()