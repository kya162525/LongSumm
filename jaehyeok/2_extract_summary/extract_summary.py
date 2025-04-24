import json
import os
import nltk
from nltk.tokenize import sent_tokenize
import logging
from datetime import datetime
import argparse
from typing import List, Dict, Tuple, Optional
from rouge_score import rouge_scorer
from tqdm import tqdm
import re

# Create logs directory if it doesn't exist
os.makedirs("./jaehyeok/logs", exist_ok=True)

# Configure logging with timestamp in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./jaehyeok/logs/extract_summary_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Download NLTK data for sentence tokenization if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logging.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

def extract_first_sentences(sections: List[Dict], num_sentences: int = 1) -> str:
    """Extract the first n sentences from each section."""
    logging.debug(f"Extracting first {num_sentences} sentences from each section")
    extracted_sentences = []
    
    for section in sections:
        if 'text' not in section or not section['text'].strip():
            continue
            
        text = section['text'].strip()
        sentences = sent_tokenize(text)
        
        if sentences:
            # Take the first n sentences (or all if there are fewer)
            section_sentences = sentences[:min(num_sentences, len(sentences))]
            extracted_sentences.extend(section_sentences)
    
    return ' '.join(extracted_sentences)

def extract_last_sentences(sections: List[Dict], num_sentences: int = 1) -> str:
    """Extract the last n sentences from each section."""
    logging.debug(f"Extracting last {num_sentences} sentences from each section")
    extracted_sentences = []
    
    for section in sections:
        if 'text' not in section or not section['text'].strip():
            continue
            
        text = section['text'].strip()
        sentences = sent_tokenize(text)
        
        if sentences:
            # Take the last n sentences (or all if there are fewer)
            section_sentences = sentences[max(0, len(sentences) - num_sentences):]
            extracted_sentences.extend(section_sentences)
    
    return ' '.join(extracted_sentences)

def extract_by_heading(sections: List[Dict], keyword: str) -> str:
    """Extract the text from sections with headings containing the specified keyword (case-insensitive)."""
    logging.debug(f"Extracting sections with headings containing '{keyword}'")
    extracted_text = []
    
    for section in sections:
        if 'heading' not in section or not section['heading'] or 'text' not in section or not section['text'].strip():
            continue
            
        # Case-insensitive search for keyword in heading
        if re.search(keyword, section['heading'], re.IGNORECASE):
            text = section['text'].strip()
            extracted_text.append(text)
    
    return ' '.join(extracted_text)

def evaluate_summary(generated_summary: str, gold_summary: str) -> Dict:
    """Evaluate the generated summary against the gold summary using ROUGE."""
    logging.debug("Calculating ROUGE scores")
    rouge_score = {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(gold_summary, generated_summary)
    
    rouge_score["rouge1"] = scores["rouge1"].fmeasure
    rouge_score["rouge2"] = scores["rouge2"].fmeasure
    rouge_score["rougeL"] = scores["rougeL"].fmeasure
    
    return rouge_score

def process_papers(method: str, num_sentences: int = 1, keyword: str = None) -> Dict:
    """Process all papers and generate summaries based on the specified method."""
    if method in ["first", "last"]:
        logging.info(f"Starting to process papers using {method} method with {num_sentences} sentences")
    else:
        logging.info(f"Starting to process papers using heading extraction with keyword '{keyword}'")
    
    # Path to JSON files
    json_path = "./papers/postprocessed/jsons"
    
    # Load gold summaries
    logging.info("Loading gold summaries")
    gold_summaries = json.load(open("./abstractive_summaries/id_summary_map.json", "r"))
    
    # Results dictionary
    results = {}
    
    # Try to load existing results, create new file if doesn't exist
    if method in ["first", "last"]:
        result_filename = f"./jaehyeok/results/summarization_{method}_{num_sentences}.json"
    else:
        result_filename = f"./jaehyeok/results/summarization_heading_{keyword}.json"
        
    try:
        logging.info(f"Loading existing results file: {result_filename}")
        results = json.load(open(result_filename, "r"))
        logging.info(f"Loaded {len(results)} existing results")
    except (FileNotFoundError, json.JSONDecodeError):
        logging.info("No existing results file found, creating new one")
        results = {}
    
    # Get list of JSON files
    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    logging.info(f"Found {len(json_files)} JSON files to process")

    # Load valid IDs from JSON file
    
    with open("/Users/jaehyeoklee/git/LongSumm/valid_ids.json", 'r') as f:
        valid_ids = json.load(f)

    for json_file in tqdm(json_files, desc="Processing papers"):
        paper_id = json_file.split('.')[0]
        
        # Skip if paper_id is not in valid_ids
        if paper_id not in valid_ids:
            logging.info(f"Skipping {paper_id} - not in valid_ids list")
            continue

        # Skip if already processed
        if paper_id in results:
            logging.info(f"Skipping {paper_id} - already processed")
            continue
        
        # Skip if no gold summary
        if paper_id not in gold_summaries:
            logging.warning(f"No gold summary found for {paper_id}. Skipping...")
            continue
        
        try:
            # Load paper
            with open(os.path.join(json_path, json_file), 'r') as f:
                paper = json.load(f)
            
            if 'sections' not in paper or not paper['sections']:
                logging.warning(f"No sections found in {paper_id}. Skipping...")
                continue
            
            # Generate summary based on method
            if method == "first":
                summary = extract_first_sentences(paper['sections'], num_sentences)
            elif method == "last":
                summary = extract_last_sentences(paper['sections'], num_sentences)
            elif method == "heading":
                summary = extract_by_heading(paper['sections'], keyword)
            else:
                logging.error(f"Unknown method: {method}")
                continue
            
            # Skip if no summary could be generated
            if not summary:
                logging.warning(f"No summary could be generated for {paper_id} using method {method}. Skipping...")
                continue
            
            # Evaluate summary
            gold_summary = gold_summaries[paper_id]
            rouge_scores = evaluate_summary(summary, gold_summary)
            
            # Save results
            results[paper_id] = {
                "gold_summary": gold_summary,
                "generated_summary": summary,
                "rouge_score": rouge_scores
            }
            
            # Save after each paper to prevent data loss
            with open(result_filename, "w") as f:
                json.dump(results, f, indent=4)
            
            logging.info(f"Successfully processed and saved results for {paper_id}")
            
        except Exception as e:
            logging.error(f"Error processing document {paper_id}: {str(e)}")
            logging.error("Continuing with next document...")
            continue
    
    logging.info(f"Completed processing papers using {method} method")
    return results

def calculate_average_scores(results: Dict) -> Dict:
    """Calculate the average ROUGE scores across all papers."""
    logging.info("Calculating average ROUGE scores")
    
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
    parser = argparse.ArgumentParser(description="Extract summaries from paper sections")
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["first", "last", "heading"], 
        default="first",
        help="Method to extract sentences: 'first' for first sentences, 'last' for last sentences, 'heading' for sections with specific headings"
    )
    parser.add_argument(
        "--num_sentences", 
        type=int, 
        default=1,
        help="Number of sentences to extract from each section (for 'first' and 'last' methods)"
    )
    parser.add_argument(
        "--keyword", 
        type=str, 
        default="introduction",
        help="Keyword to search for in section headings (for 'heading' method)"
    )
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs("./jaehyeok/results", exist_ok=True)
    
    # Process papers
    if args.method in ["first", "last"]:
        results = process_papers(args.method, args.num_sentences)
    else:  # heading method
        results = process_papers(args.method, keyword=args.keyword)
    
    # Calculate average scores
    avg_scores = calculate_average_scores(results)
    
    # Save all results
    if args.method in ["first", "last"]:
        result_filename = f"./jaehyeok/results/all_results_{args.method}_{args.num_sentences}.json"
    else:
        result_filename = f"./jaehyeok/results/all_results_heading_{args.keyword}.json"
        
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Saved all evaluation results to {result_filename}")
    
    # Also save average scores separately
    if args.method in ["first", "last"]:
        avg_filename = f"./jaehyeok/results/avg_scores_{args.method}_{args.num_sentences}.json"
    else:
        avg_filename = f"./jaehyeok/results/avg_scores_heading_{args.keyword}.json"
        
    with open(avg_filename, "w") as f:
        json.dump(avg_scores, f, indent=4)
    
    logging.info(f"Extraction process completed using {args.method} method")
    if args.method in ["first", "last"]:
        logging.info(f"Used {args.num_sentences} sentences per section")
    else:
        logging.info(f"Extracted sections with headings containing '{args.keyword}'")
    logging.info(f"Average ROUGE scores: {avg_scores}")
    logging.info(f"All individual paper results saved to {result_filename}")

if __name__ == "__main__":
    main()