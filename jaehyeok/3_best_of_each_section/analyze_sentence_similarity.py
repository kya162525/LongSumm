import json
import os
import logging
import argparse
import nltk
from datetime import datetime
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np

nltk.download('punkt_tab')

# Setup logging
os.makedirs("./jaehyeok/logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./jaehyeok/logs/sentence_similarity_analysis_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def load_papers(json_path: str) -> Dict:
    """Load all papers from JSON files."""
    logging.info(f"Loading papers from {json_path}")
    papers = {}

    # Load valid paper IDs
    logging.info("Loading valid paper IDs")
    try:
        with open("./valid_ids.json", 'r') as f:
            valid_ids = json.load(f)
        logging.info(f"Loaded {len(valid_ids)} valid paper IDs")
    except Exception as e:
        logging.error(f"Error loading valid paper IDs: {str(e)}")
        valid_ids = []
    
    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
    logging.info(f"Found {len(json_files)} JSON files")
    
    for json_file in tqdm(json_files, desc="Loading papers"):
        paper_id = json_file.split('.')[0]
        if valid_ids and paper_id not in valid_ids:
            continue
        try:
            with open(os.path.join(json_path, json_file), 'r') as f:
                paper = json.load(f)
            papers[paper_id] = paper
            logging.debug(f"Loaded paper {paper_id}")
        except Exception as e:
            logging.error(f"Error loading paper {paper_id}: {str(e)}")
    logging.info(f"Loaded {len(papers)} papers")
    return papers

def load_gold_summaries(summaries_path: str) -> Dict:
    """Load gold summaries."""
    logging.info(f"Loading gold summaries from {summaries_path}")
    summaries = {}
    
    try:
        with open(summaries_path, 'r') as f:
            summaries = json.load(f)
        logging.info(f"Loaded {len(summaries)} gold summaries")
    except Exception as e:
        logging.error(f"Error loading gold summaries: {str(e)}")
    
    return summaries

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    if not text or not text.strip():
        return []
    sentences = nltk.sent_tokenize(text.strip())
    return [sent.strip() for sent in sentences if sent.strip()]

def calculate_sentence_similarity(model, section_sentences: List[str], summary_sentences: List[str]) -> List[Dict]:
    """Calculate similarity between all section sentences and summary sentences."""
    # Return early if either list is empty
    if not section_sentences or not summary_sentences:
        return []
    
    # Encode all sentences
    section_embeddings = model.encode(section_sentences, convert_to_tensor=True)
    summary_embeddings = model.encode(summary_sentences, convert_to_tensor=True)
    
    # Calculate similarity matrix (section_sentences x summary_sentences)
    similarity_matrix = util.pytorch_cos_sim(section_embeddings, summary_embeddings).cpu().numpy()
    
    # For each section sentence, find the highest similarity with any summary sentence
    best_matches = []
    for i, section_sent in enumerate(section_sentences):
        # Find index of max similarity for this section sentence
        best_summary_idx = int(np.argmax(similarity_matrix[i]))
        best_similarity = similarity_matrix[i][best_summary_idx]
        
        best_matches.append({
            "section_sentence_idx": i,
            "section_sentence": section_sent,
            "summary_sentence_idx": best_summary_idx,
            "summary_sentence": summary_sentences[best_summary_idx],
            "similarity_score": float(best_similarity)
        })
    
    # Sort by similarity score (highest first)
    best_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return best_matches

def analyze_paper_sentence_similarity(paper: Dict, gold_summary: str, model: SentenceTransformer) -> Dict:
    """Analyze sentence-level similarity between gold summary and each section of a paper."""
    results = {
        "summary_sentences": [],
        "section_results": []
    }
    
    # Split gold summary into sentences
    summary_sentences = split_into_sentences(gold_summary)
    results["summary_sentences"] = summary_sentences
    
    if 'sections' not in paper or not paper['sections']:
        logging.warning(f"No sections found in paper")
        return results
    
    # Process each section
    for i, section in enumerate(paper['sections']):
        if 'text' not in section or not section['text'].strip():
            continue
            
        section_text = section['text'].strip()
        heading = section.get('heading', f"Section {i+1}")
        section_sentences = split_into_sentences(section_text)
        sentence_matches = calculate_sentence_similarity(model, section_sentences, summary_sentences)
        
        section_result = {
            "section_idx": i,
            "heading": heading,
            "num_sentences": len(section_sentences),
            "sentence_matches": sentence_matches
        }
        
        results["section_results"].append(section_result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze sentence-level similarity between gold summaries and paper sections")
    parser.add_argument(
        "--papers_path", 
        type=str, 
        default="./papers/json_files",
        help="Path to the paper JSON files"
    )
    parser.add_argument(
        "--summaries_path", 
        type=str, 
        default="./abstractive_summaries/id_summary_map.json",
        help="Path to the gold summaries file"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./jaehyeok/results/sentence_similarity_analysis.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of papers to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load model
    logging.info(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    
    # Load papers and gold summaries
    papers = load_papers(args.papers_path)
    
    # Try to load existing results, create new file if doesn't exist
    try:
        logging.info("Loading existing results file")
        with open(args.output_path, "r") as f:
            results = json.load(f)
        paper_results = results.get("paper_results", {})
        logging.info(f"Loaded {len(paper_results)} existing results")
    except (FileNotFoundError, json.JSONDecodeError):
        logging.info("No existing results file found, creating new one")
        paper_results = {}

    logging.info("Loading gold summaries")
    gold_summaries = load_gold_summaries(args.summaries_path)
    
    # Process papers
    paper_ids = list(papers.keys())
    if args.limit:
        paper_ids = paper_ids[:args.limit]
        
    for paper_id in tqdm(paper_ids, desc="Processing papers"):
        paper_data = papers[paper_id]

        if paper_id in paper_results:
            logging.info(f"Skipping {paper_id} - already processed")
            continue

        gold_summary = gold_summaries.get(paper_id, None)
        if gold_summary is None:
            logging.warning(f"Skipping {paper_id} - no gold summary found")
            continue

        try:
            paper_result = analyze_paper_sentence_similarity(paper_data, gold_summary, model)
            paper_results[paper_id] = paper_result
            
            # Log summary of findings
            total_sections = len(paper_result["section_results"])
            total_summary_sentences = len(paper_result["summary_sentences"]) 
            logging.info(f"Paper {paper_id}: analyzed {total_sections} sections with {total_summary_sentences} summary sentences")
            
            # # Save incremental results to avoid losing data in case of errors
            # if len(paper_results) % 10 == 0:
            #     final_results = {
            #         "paper_results": paper_results,
            #         "metadata": {
            #             "model": args.model_name,
            #             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            #             "papers_processed": len(paper_results)
            #         }
            #     }
            #     with open(args.output_path, 'w') as f:
            #         json.dump(final_results, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error analyzing paper {paper_id}: {str(e)}")
    
    # Compile stats on sentence position distribution
    sentence_position_stats = {}
    for paper_id, paper_result in paper_results.items():
        for section_result in paper_result["section_results"]:
            if not section_result["sentence_matches"]:
                continue
                
            # Consider only the top match for each section
            top_match = section_result["sentence_matches"][0]
            section_len = section_result["num_sentences"]
            
            # Skip if section has no sentences
            if section_len == 0:
                continue
                
            # Calculate relative position (0-1 range)
            relative_pos = top_match["section_sentence_idx"] / section_len
            
            # Bin into 10 percentile ranges
            percentile_bin = int(relative_pos * 10)
            if percentile_bin == 10:  # Handle edge case for 1.0
                percentile_bin = 9
                
            # Update stats
            if percentile_bin not in sentence_position_stats:
                sentence_position_stats[percentile_bin] = 0
            sentence_position_stats[percentile_bin] += 1
    
    # Save final results
    final_results = {
        "paper_results": paper_results,
        "stats": {
            "papers_processed": len(paper_results),
            "sentence_position_distribution": sentence_position_stats
        },
        "metadata": {
            "model": args.model_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    logging.info(f"Saving results to {args.output_path}")
    with open(args.output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logging.info(f"Analysis complete. Processed {len(paper_results)} papers.")
    
    # Print summary of sentence position distribution
    logging.info("Sentence Position Distribution (by percentile bin):")
    total_matches = sum(sentence_position_stats.values())
    for bin_idx in range(10):
        count = sentence_position_stats.get(bin_idx, 0)
        percentage = (count / total_matches * 100) if total_matches > 0 else 0
        logging.info(f"  {bin_idx*10}-{(bin_idx+1)*10}%: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()