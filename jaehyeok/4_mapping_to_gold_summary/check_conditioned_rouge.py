import json
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
from rouge_score import rouge_scorer

# Setup logging
os.makedirs("./jaehyeok/logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./jaehyeok/logs/conditional_rouge_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def load_mapping_data(path: str) -> Dict:
    """Load the existing mapping data from the specified JSON file."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded mapping data for {len(data)} papers")
        return data
    except Exception as e:
        logging.error(f"Error loading mapping data: {str(e)}")
        return {}

def calculate_rouge_scores(reference: str, candidate: str) -> Dict:
    """Calculate ROUGE scores between reference and candidate summaries."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    
    return {
        "rouge1": {
            "precision": float(scores["rouge1"].precision),
            "recall": float(scores["rouge1"].recall),
            "fmeasure": float(scores["rouge1"].fmeasure)
        },
        "rouge2": {
            "precision": float(scores["rouge2"].precision),
            "recall": float(scores["rouge2"].recall),
            "fmeasure": float(scores["rouge2"].fmeasure)
        },
        "rougeL": {
            "precision": float(scores["rougeL"].precision),
            "recall": float(scores["rougeL"].recall),
            "fmeasure": float(scores["rougeL"].fmeasure)
        }
    }

def filter_mappings_by_similarity(mappings: List[Dict], threshold: float) -> List[Dict]:
    """Filter mappings based on similarity threshold."""
    return [m for m in mappings if m["similarity"] >= threshold]

def create_extractive_summary(mappings: List[Dict]) -> str:
    """Create an extractive summary from the mappings."""
    if not mappings:
        return ""
    
    # Sort the mappings by summary index to maintain the original order
    sorted_mappings = sorted(mappings, key=lambda x: x["summary_idx"])
    
    # Extract the paper sentences
    extractive_sentences = [mapping["paper_sentence"] for mapping in sorted_mappings]
    
    # Join the sentences to form the summary
    return " ".join(extractive_sentences)

def evaluate_with_conditions(mapping_data: Dict, similarity_threshold: float, output_path: str):
    """Evaluate ROUGE scores with conditions applied to the mappings."""
    results = {}
    
    for paper_id, paper_data in tqdm(mapping_data.items(), desc="Processing papers"):
        if "mappings" not in paper_data or not paper_data["mappings"]:
            logging.warning(f"Skipping {paper_id} - no mappings found")
            continue
            
        gold_summary = paper_data.get("gold_summary", "")
        if not gold_summary:
            logging.warning(f"Skipping {paper_id} - no gold summary found")
            continue
            
        try:
            # Filter mappings based on similarity threshold
            filtered_mappings = filter_mappings_by_similarity(paper_data["mappings"], similarity_threshold)
            
            # If no mappings meet the threshold, skip this paper
            skipped_papers_count = 0  # This should be defined at the beginning of the function
            if not filtered_mappings:
                logging.info(f"Skipping {paper_id} - no mappings meet similarity threshold {similarity_threshold}")
                skipped_papers_count += 1
                continue
                
            # Create extractive summary from filtered mappings
            extractive_summary = create_extractive_summary(filtered_mappings)
            
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(gold_summary, extractive_summary)
            
            logging.info(f"Paper {paper_id} ROUGE-1 F1: {rouge_scores['rouge1']['fmeasure']:.4f}")
            logging.info(f"Paper {paper_id} ROUGE-2 F1: {rouge_scores['rouge2']['fmeasure']:.4f}")
            logging.info(f"Paper {paper_id} ROUGE-L F1: {rouge_scores['rougeL']['fmeasure']:.4f}")
            
            # Record results
            results[paper_id] = {
                "paper_id": paper_id,
                "gold_summary": gold_summary,
                "extractive_summary": extractive_summary,
                "filtered_mappings": filtered_mappings,
                "rouge_scores": rouge_scores,
                "similarity_threshold": similarity_threshold,
                "stats": {
                    "num_original_mappings": len(paper_data["mappings"]),
                    "num_filtered_mappings": len(filtered_mappings),
                    "avg_similarity": sum(m["similarity"] for m in filtered_mappings) / len(filtered_mappings) if filtered_mappings else 0
                }
            }
            
        except Exception as e:
            logging.error(f"Error processing paper {paper_id}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Calculate overall statistics
    if results:
        rouge1_f1 = [paper_result["rouge_scores"]["rouge1"]["fmeasure"] for paper_result in results.values()]
        rouge2_f1 = [paper_result["rouge_scores"]["rouge2"]["fmeasure"] for paper_result in results.values()]
        rougeL_f1 = [paper_result["rouge_scores"]["rougeL"]["fmeasure"] for paper_result in results.values()]
        
        avg_rouge1 = sum(rouge1_f1) / len(rouge1_f1) if rouge1_f1 else 0
        avg_rouge2 = sum(rouge2_f1) / len(rouge2_f1) if rouge2_f1 else 0
        avg_rougeL = sum(rougeL_f1) / len(rougeL_f1) if rougeL_f1 else 0
        
        original_mappings = sum(paper_result["stats"]["num_original_mappings"] for paper_result in results.values())
        filtered_mappings = sum(paper_result["stats"]["num_filtered_mappings"] for paper_result in results.values())
        
        logging.info(f"Overall statistics:")
        logging.info(f"Papers processed: {len(results)}")
        logging.info(f"Similarity threshold: {similarity_threshold}")
        logging.info(f"Original mappings: {original_mappings}")
        logging.info(f"Filtered mappings: {filtered_mappings}")
        logging.info(f"Percentage of mappings retained: {filtered_mappings/original_mappings*100:.2f}%")
        logging.info(f"Average ROUGE-1 F1: {avg_rouge1:.4f}")
        logging.info(f"Average ROUGE-2 F1: {avg_rouge2:.4f}")
        logging.info(f"Average ROUGE-L F1: {avg_rougeL:.4f}")
        logging.info(f"Skipped papers due to no mappings meeting threshold: {skipped_papers_count}")
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate ROUGE scores with conditions on existing mapping data")
    parser.add_argument(
        "--mapping_path", 
        type=str, 
        default="./jaehyeok/datasets/mapping_gold_summary_for_training_sets.json",
        help="Path to the mapping data file"
    )
    parser.add_argument(
        "--similarity_threshold", 
        type=float, 
        default=0.581,
        help="Minimum similarity threshold for mappings"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=f"./jaehyeok/datasets/conditional_rouge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Path to save results"
    )
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load mapping data
    mapping_data = load_mapping_data(args.mapping_path)
    
    # Evaluate with conditions
    evaluate_with_conditions(mapping_data, args.similarity_threshold, args.output_path)
    
    logging.info(f"Processing complete. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()