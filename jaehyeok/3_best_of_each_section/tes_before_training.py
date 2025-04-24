import json
import os
import logging
from datetime import datetime
import argparse
from typing import List, Dict, Any
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np

# Create logs directory if it doesn't exist
os.makedirs("./logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./logs/test_before_training_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_gold_summaries(file_path: str) -> Dict[str, str]:
    """Load gold summaries from the provided file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_summary(generated_summary: str, gold_summary: str) -> Dict:
    """Evaluate the generated summary against the gold summary using ROUGE."""
    rouge_score = {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(gold_summary, generated_summary)
    
    rouge_score["rouge1"] = scores["rouge1"].fmeasure
    rouge_score["rouge2"] = scores["rouge2"].fmeasure
    rouge_score["rougeL"] = scores["rougeL"].fmeasure
    
    return rouge_score

def generate_summaries(data: List[Dict[str, Any]], similarity_threshold: float = 0.5) -> Dict[str, str]:
    """
    Generate summaries by grouping entries by paper_id, sorting by section_id,
    and concatenating sentences with similarity above the threshold.
    """
    summaries = {}
    
    # Group data by paper_id
    paper_groups = {}
    for item in data:
        paper_id = item.get('paper_id')
        if paper_id:
            if paper_id not in paper_groups:
                paper_groups[paper_id] = []
            paper_groups[paper_id].append(item)
    
    # Process each paper
    for paper_id, items in paper_groups.items():
        # Sort by section_id
        sorted_items = sorted(items, key=lambda x: x.get('section_id', 0))
        
        # Filter items with similarity >= threshold and collect best sentences
        summary_sentences = []
        for item in sorted_items:
            if item.get('similarity', 0) >= similarity_threshold:
                if 'best_sentence' in item:
                    summary_sentences.append(item['best_sentence'])
        
        # Create summary by joining sentences
        summary = ' '.join(summary_sentences)
        summaries[paper_id] = summary
    
    return summaries

def calculate_average_scores(results: Dict) -> Dict:
    """Calculate the average ROUGE scores across all papers."""
    if not results:
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    
    total_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    count = 0
    
    for paper_id, paper_result in results.items():
        if "rouge_score" in paper_result:
            total_scores["rouge1"] += paper_result["rouge_score"]["rouge1"]
            total_scores["rouge2"] += paper_result["rouge_score"]["rouge2"]
            total_scores["rougeL"] += paper_result["rouge_score"]["rougeL"]
            count += 1
    
    if count == 0:
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    
    avg_scores = {
        "rouge1": total_scores["rouge1"] / count,
        "rouge2": total_scores["rouge2"] / count,
        "rougeL": total_scores["rougeL"] / count
    }
    
    logging.info(f"Average ROUGE scores across {count} papers: {avg_scores}")
    return avg_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated summaries against gold summaries")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="./jaehyeok/datasets/pairs_datasets.jsonl",
        help="Path to input JSONL file containing entries with paper_id, section_id, and similarity"
    )
    parser.add_argument(
        "--gold_summaries", 
        type=str, 
        default="./abstractive_summaries/id_summary_map.json",
        help="Path to the JSON file containing gold summaries"
    )
    parser.add_argument(
        "--similarity_threshold", 
        type=float, 
        default=0.5,
        help="Minimum similarity threshold for sentences to include in summary"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logging.info(f"Loading data from {args.input_file}")
    data = load_jsonl(args.input_file)
    logging.info(f"Loaded {len(data)} entries")
    
    # Load gold summaries
    logging.info(f"Loading gold summaries from {args.gold_summaries}")
    gold_summaries = load_gold_summaries(args.gold_summaries)
    logging.info(f"Loaded {len(gold_summaries)} gold summaries")
    
    # Generate summaries
    logging.info(f"Generating summaries with similarity threshold: {args.similarity_threshold}")
    generated_summaries = generate_summaries(data, args.similarity_threshold)
    logging.info(f"Generated {len(generated_summaries)} summaries")
    
    # Evaluate summaries
    results = {}
    logging.info("Evaluating summaries")
    
    for paper_id, summary in tqdm(generated_summaries.items(), desc="Evaluating"):
        if paper_id in gold_summaries:
            gold_summary = gold_summaries[paper_id]
            rouge_scores = evaluate_summary(summary, gold_summary)
            
            results[paper_id] = {
                "generated_summary": summary,
                "gold_summary": gold_summary,
                "rouge_score": rouge_scores
            }
        else:
            logging.warning(f"No gold summary found for paper_id: {paper_id}")
    
    # Calculate average scores
    avg_scores = calculate_average_scores(results)
    
    # Save results
    result_filename = os.path.join(args.output_dir, f"evaluation_results_thresh_{args.similarity_threshold}.json")
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Saved individual results to {result_filename}")
    
    # Save average scores
    avg_scores_filename = os.path.join(args.output_dir, f"average_scores_thresh_{args.similarity_threshold}.json")
    with open(avg_scores_filename, "w") as f:
        json.dump(avg_scores, f, indent=4)
    logging.info(f"Saved average scores to {avg_scores_filename}")
    
    # Print summary of results
    logging.info(f"Evaluation complete. Average ROUGE scores: {avg_scores}")
    print(f"Average ROUGE scores: R1={avg_scores['rouge1']:.4f}, R2={avg_scores['rouge2']:.4f}, RL={avg_scores['rougeL']:.4f}")

if __name__ == "__main__":
    main()