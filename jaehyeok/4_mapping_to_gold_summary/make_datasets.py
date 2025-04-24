import json
import os
import nltk
from nltk.tokenize import sent_tokenize
import logging
from datetime import datetime
import argparse
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
# Add ROUGE score calculation imports
from rouge_score import rouge_scorer

# Setup logging
os.makedirs("./jaehyeok/logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./jaehyeok/logs/make_datasets_{timestamp}.log"

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

def load_papers(path: str) -> Dict[str, str]:
    """Load all paper texts from the specified directory."""
    logging.info(f"Loading papers from {path}")
    papers = {}
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    logging.info(f"Found {len(files)} paper files")
    
    for file in tqdm(files, desc="Loading papers"):
        paper_id = file.split(".")[0]
        try:
            with open(os.path.join(path, file), "r") as f:
                papers[paper_id] = f.read()
            logging.debug(f"Loaded paper {paper_id}")
        except Exception as e:
            logging.error(f"Error loading paper {paper_id}: {str(e)}")
    
    logging.info(f"Successfully loaded {len(papers)} papers")
    return papers

def load_gold_summaries(summaries_path: str) -> Dict[str, str]:
    """Load gold summaries."""
    logging.info(f"Loading gold summaries from {summaries_path}")
    try:
        with open(summaries_path, 'r') as f:
            summaries = json.load(f)
        logging.info(f"Loaded {len(summaries)} gold summaries")
        return summaries
    except Exception as e:
        logging.error(f"Error loading gold summaries: {str(e)}")
        return {}

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

def find_most_similar_sentences(
    summary_sentences: List[str], 
    paper_sentences: List[str], 
    model: SentenceTransformer
) -> List[Dict]:
    """For each summary sentence, find the most similar paper sentence."""
    mappings = []
    
    # Encode all paper sentences once for efficiency
    logging.info(f"Encoding {len(paper_sentences)} paper sentences")
    paper_embeddings = model.encode(paper_sentences, convert_to_tensor=True)
    
    for i, summary_sentence in enumerate(tqdm(summary_sentences, desc="Finding similar sentences")):
        # Encode the summary sentence
        summary_embedding = model.encode(summary_sentence, convert_to_tensor=True)
        
        # Calculate similarities with all paper sentences
        similarities = util.pytorch_cos_sim(summary_embedding, paper_embeddings)[0]
        
        # Find the most similar paper sentence
        max_idx = similarities.argmax().item()
        max_similarity = similarities[max_idx].item()
        
        mapping = {
            "summary_idx": i,
            "summary_sentence": summary_sentence,
            "paper_idx": max_idx,
            "paper_sentence": paper_sentences[max_idx],
            "similarity": max_similarity
        }
        mappings.append(mapping)
    
    return mappings

def create_extractive_summary(mappings: List[Dict]) -> str:
    """Create an extractive summary from the mappings."""
    # Sort the mappings by summary index to maintain the original order
    sorted_mappings = sorted(mappings, key=lambda x: x["summary_idx"])
    
    # Extract the paper sentences
    extractive_sentences = [mapping["paper_sentence"] for mapping in sorted_mappings]
    
    # Join the sentences to form the summary
    return " ".join(extractive_sentences)

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

def main():
    parser = argparse.ArgumentParser(description="Create extractive summaries by mapping gold summary sentences to paper sentences")
    parser.add_argument(
        "--papers_path", 
        type=str, 
        default="./papers/postprocessed/full_texts/",
        help="Path to the paper text files"
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
        default="./jaehyeok/datasets/mapping_gold_summary_for_training_sets.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding calculation"
    )
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load model
    logging.info(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    
    # Load papers and gold summaries
    papers = load_papers(args.papers_path)
    gold_summaries = load_gold_summaries(args.summaries_path)
    
    # Try to load existing results
    try:
        logging.info(f"Loading existing results from {args.output_path}")
        with open(args.output_path, "r") as f:
            results = json.load(f)
        logging.info(f"Loaded {len(results)} existing results")
    except (FileNotFoundError, json.JSONDecodeError):
        logging.info("No existing results file found, creating new one")
        results = {}
    with open("./valid_ids.json", 'r') as f:
        valid_ids = json.load(f)
    logging.info(f"Loaded {len(valid_ids)} valid paper IDs")

    # Process each paper
    for paper_id, paper_text in tqdm(papers.items(), desc="Processing papers"):
        if paper_id not in valid_ids:
            logging.info(f"Skipping {paper_id} - not in valid IDs")
            continue
        if paper_id in results:
            logging.info(f"Skipping {paper_id} - already processed")
            continue
        gold_summary = gold_summaries.get(paper_id, None)
        if gold_summary is None:
            logging.warning(f"Skipping {paper_id} - no gold summary found")
            continue
            
        try:
            logging.info(f"Processing paper {paper_id}")
            summary_sentences = split_into_sentences(gold_summary)
            paper_sentences = split_into_sentences(paper_text)
            
            logging.info(f"Paper {paper_id}: {len(paper_sentences)} sentences, Summary: {len(summary_sentences)} sentences")
            
            mappings = find_most_similar_sentences(summary_sentences, paper_sentences, model)
            extractive_summary = create_extractive_summary(mappings)
            
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(gold_summary, extractive_summary)
            logging.info(f"Paper {paper_id} ROUGE-1 F1: {rouge_scores['rouge1']['fmeasure']:.4f}")
            logging.info(f"Paper {paper_id} ROUGE-2 F1: {rouge_scores['rouge2']['fmeasure']:.4f}")
            logging.info(f"Paper {paper_id} ROUGE-L F1: {rouge_scores['rougeL']['fmeasure']:.4f}")
            
            results[paper_id] = {
                "paper_id": paper_id,
                "gold_summary": gold_summary,
                "extractive_summary": extractive_summary,
                "mappings": mappings,
                "rouge_scores": rouge_scores,  # Add ROUGE scores to results
                "stats": {
                    "num_summary_sentences": len(summary_sentences),
                    "num_paper_sentences": len(paper_sentences),
                    "avg_similarity": sum(m["similarity"] for m in mappings) / len(mappings) if mappings else 0
                }
            }
            
            # Save results after each paper
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=4)
                
            logging.info(f"Successfully processed paper {paper_id}")
            
        except Exception as e:
            logging.error(f"Error processing paper {paper_id}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Calculate overall statistics
    if results:
        avg_similarities = [paper_result["stats"]["avg_similarity"] for paper_result in results.values()]
        overall_avg_similarity = sum(avg_similarities) / len(avg_similarities)
        
        total_summary_sentences = sum(paper_result["stats"]["num_summary_sentences"] for paper_result in results.values())
        total_paper_sentences = sum(paper_result["stats"]["num_paper_sentences"] for paper_result in results.values())
        
        # Calculate average ROUGE scores
        rouge1_f1 = [paper_result["rouge_scores"]["rouge1"]["fmeasure"] for paper_result in results.values()]
        rouge2_f1 = [paper_result["rouge_scores"]["rouge2"]["fmeasure"] for paper_result in results.values()]
        rougeL_f1 = [paper_result["rouge_scores"]["rougeL"]["fmeasure"] for paper_result in results.values()]
        
        avg_rouge1 = sum(rouge1_f1) / len(rouge1_f1) if rouge1_f1 else 0
        avg_rouge2 = sum(rouge2_f1) / len(rouge2_f1) if rouge2_f1 else 0
        avg_rougeL = sum(rougeL_f1) / len(rougeL_f1) if rougeL_f1 else 0
        
        logging.info(f"Overall statistics:")
        logging.info(f"Papers processed: {len(results)}")
        logging.info(f"Total summary sentences: {total_summary_sentences}")
        logging.info(f"Total paper sentences: {total_paper_sentences}")
        logging.info(f"Average similarity: {overall_avg_similarity:.4f}")
        logging.info(f"Average ROUGE-1 F1: {avg_rouge1:.4f}")
        logging.info(f"Average ROUGE-2 F1: {avg_rouge2:.4f}")
        logging.info(f"Average ROUGE-L F1: {avg_rougeL:.4f}")
    
    logging.info(f"Processing complete. Processed {len(results)} papers.")

if __name__ == "__main__":
    main()