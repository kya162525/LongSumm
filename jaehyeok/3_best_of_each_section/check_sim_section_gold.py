import json
import os
import logging
from datetime import datetime
import argparse
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Setup logging
os.makedirs("./jaehyeok/logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./jaehyeok/logs/similarity_analysis_{timestamp}.log"

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
        if paper_id not in valid_ids:
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

# def truncate_text(text: str, max_length: int = 2048) -> str:
#     """Truncate text to handle token limits."""
#     words = text.split()
#     if len(words) <= max_length:
#         return text
#     return ' '.join(words[:max_length])

def calculate_similarity(model: SentenceTransformer, text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts using miniLM."""
    # Truncate texts to avoid token limits
    # text1 = truncate_text(text1)
    # text2 = truncate_text(text2)
    text1 = text1.strip()
    text2 = text2.strip()
    
    # Encode texts
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    
    return cosine_score

def analyze_paper_similarity(paper: Dict, gold_summary: str, model: SentenceTransformer) -> Dict:
    """Analyze similarity between gold summary and each section of a paper."""
    results = {
        "overall_similarity": 0,
        "section_similarities": [],
        "max_similarity_section": {"heading": "", "similarity": 0, "index": -1},
        "avg_similarity": 0
    }
    
    if 'sections' not in paper or not paper['sections']:
        logging.warning(f"No sections found in paper")
        return results
    
    # Calculate similarity for each section
    section_similarities = []
    for i, section in enumerate(paper['sections']):
        if 'text' not in section or not section['text'].strip():
            continue
            
        text = section['text'].strip()
        heading = section.get('heading', f"Section {i+1}")
        
        sim_score = calculate_similarity(model, gold_summary, text)
        
        section_result = {
            "heading": heading,
            "similarity": sim_score,
            "length": len(text),
            "index": i
        }
        results["section_similarities"].append(section_result)
        section_similarities.append(sim_score)
        
        # Update max similarity section
        if sim_score > results["max_similarity_section"]["similarity"]:
            results["max_similarity_section"] = {
                "heading": heading,
                "similarity": sim_score,
                "index": i
            }
    
    # Calculate average similarity
    if section_similarities:
        results["avg_similarity"] = sum(section_similarities) / len(section_similarities)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze similarity between gold summaries and paper sections using miniLM")
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
        default="./jaehyeok/results/similarity_analysis.json",
        help="Path to save results"
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
        id_score_map = json.load(open(args.output_path, "r"))
        logging.info(f"Loaded {len(id_score_map)} existing results")
    except (FileNotFoundError, json.JSONDecodeError):
        logging.info("No existing results file found, creating new one")
        id_score_map = {}

    logging.info("Loading gold summaries")
    gold_summaries = load_gold_summaries(args.summaries_path)
    
    # Calculate similarities
    for file in tqdm(papers, desc="Processing files"):
        paper_id = file
        paper_data = papers[file]

        if paper_id in id_score_map:
            logging.info(f"Skipping {paper_id} - already processed")
            continue

        gold_summary = gold_summaries.get(paper_id, None)
        if gold_summary is None:
            logging.warning(f"Skipping {paper_id} - no gold summary found")
            continue

        try:
            paper_results = analyze_paper_similarity(paper_data, gold_summary, model)
            id_score_map[paper_id] = paper_results
            logging.info(f"Paper {paper_id}: avg similarity = {paper_results['avg_similarity']:.4f} (Sections: {len(paper_results['section_similarities'])})")
        except Exception as e:
            logging.error(f"Error analyzing paper {paper_id}: {str(e)}")
    
    # Calculate aggregate statistics
    overall_similarities = [r.get("avg_similarity", 0) for r in id_score_map.values() if isinstance(r, dict)]
    avg_overall = sum(overall_similarities) / len(overall_similarities) if overall_similarities else 0
    
    final_results = {
        "paper_results": id_score_map,
        "stats": {
            "num_papers_analyzed": len(id_score_map),
            "avg_avg_similarity": avg_overall
        }
    }
    
    # Save results
    logging.info(f"Saving results to {args.output_path}")
    with open(args.output_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    logging.info(f"Analysis complete. Processed {len(id_score_map)} papers.")

if __name__ == "__main__":
    main()