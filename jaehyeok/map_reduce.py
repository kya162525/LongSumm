import json
import os
import nltk
from nltk.tokenize import sent_tokenize
import textwrap
from typing import List, Dict
import logging
from datetime import datetime

from ollama import ChatResponse, chat
from rouge_score import rouge_scorer
from tqdm import tqdm

# Create logs directory if it doesn't exist
os.makedirs("./jaehyeok/logs", exist_ok=True)

# Configure logging with timestamp in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./jaehyeok/logs/map_reduce_{timestamp}.log"

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
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logging.info("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')

def chunk_text(text: str, max_chunk_size: int = 10000) -> List[str]:
    """Split text into chunks of approximately max_chunk_size characters."""
    logging.debug(f"Starting to chunk text with max_chunk_size={max_chunk_size}")
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        # Add sentence length plus a space
        sentence_size = len(sentence) + 1
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # If adding this sentence exceeds the chunk size, finalize current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    logging.debug(f"Created {len(chunks)} chunks from text")
    return chunks

def adjust_chunk_size(text: str, initial_chunk_size: int = 10000) -> int:
    """Adjust chunk size to ensure the number of chunks is not too large."""
    logging.info(f"Adjusting chunk size, starting with initial size of {initial_chunk_size}")
    # First try with initial chunk size
    chunks = chunk_text(text, initial_chunk_size)
    
    # If number of chunks is more than 10, adjust chunk size
    if len(chunks) > 10:
        # Calculate new chunk size to target 10 chunks
        total_length = sum(len(chunk) for chunk in chunks)
        new_chunk_size = total_length // 10
        logging.info(f"Adjusted chunk size from {initial_chunk_size} to {new_chunk_size} to target 10 chunks")
        return new_chunk_size
    
    logging.info(f"Using original chunk size {initial_chunk_size}, resulted in {len(chunks)} chunks")
    return initial_chunk_size

def summarize_chunk(chunk: str, model: str = "qwen2.5:14b") -> str:
    """Summarize a single chunk of text using the specified model."""
    logging.debug(f"Summarizing chunk of size {len(chunk)} characters using model {model}")
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who summarizes scientific documents. Please provide a concise summary of the text.",
            },
            {"role": "user", "content": "Can you summarize the following text: "},
            {
                "role": "user",
                "content": chunk,
            },
        ],
    )
    logging.debug("Chunk summarization completed")
    return response.message.content

def combine_summaries(summaries: List[str], model: str = "qwen2.5:14b") -> str:
    """Combine multiple chunk summaries into a final coherent summary."""
    logging.debug(f"Combining {len(summaries)} summaries using model {model}")
    combined_text = "\n\n".join([f"Chunk {i+1}: {summary}" for i, summary in enumerate(summaries)])
    
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who combines multiple partial summaries into a coherent final summary. Avoid redundancy and ensure the final summary flows well.",
            },
            {"role": "user", "content": "Below are summaries of different parts of a scientific paper. Please combine them into a single coherent summary:"},
            {
                "role": "user",
                "content": combined_text,
            },
        ],
    )
    logging.debug("Summary combination completed")
    return response.message.content

def map_reduce_summarize(text: str, model: str = "qwen2.5:14b", initial_chunk_size: int = 10000) -> str:
    """Apply map-reduce paradigm to summarize long text."""
    logging.info("Starting map-reduce summarization")
    
    # Adjust chunk size if needed
    chunk_size = adjust_chunk_size(text, initial_chunk_size)
    
    # Map phase: split and summarize chunks
    chunks = chunk_text(text, chunk_size)
    logging.info(f"Text split into {len(chunks)} chunks (chunk size: {chunk_size})")
    
    chunk_summaries = []
    for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
        logging.info(f"Summarizing chunk {i+1}/{len(chunks)}")
        summary = summarize_chunk(chunk, model)
        chunk_summaries.append(summary)
    
    # Reduce phase: combine summaries
    if len(chunk_summaries) == 1:
        logging.info("Only one chunk summary produced, using it as final summary")
        return chunk_summaries[0]
    
    logging.info("Combining chunk summaries into final summary")
    final_summary = combine_summaries(chunk_summaries, model)
    logging.info("Map-reduce summarization completed")
    return final_summary

def main():
    logging.info("Starting map-reduce summarization process")
    path = "./papers/postprocessed/full_texts/"
    files = os.listdir(path)
    logging.info(f"Found {len(files)} files to process in {path}")

    logging.info("Loading gold summaries")
    gold_summaries = json.load(open("./abstractive_summaries/id_summary_map.json", "r"))
    
    # Try to load existing results, create new file if doesn't exist
    try:
        logging.info("Loading existing results file")
        id_score_map = json.load(open("./jaehyeok/results/summarization_map_reduce.json", "r"))
        logging.info(f"Loaded {len(id_score_map)} existing results")
    except (FileNotFoundError, json.JSONDecodeError):
        logging.info("No existing results file found, creating new one")
        id_score_map = {}

    for file in tqdm(files, desc="Processing files"):
        file_name = file.split(".")[0]

        if file_name in id_score_map:
            logging.info(f"Skipping {file_name} - already processed")
            continue

        gold_summary = gold_summaries.get(file_name, None)
        if gold_summary is None:
            logging.warning(f"No gold summary found for {file_name}. Skipping...")
            continue

        try:
            logging.info(f"Processing document: {file_name}")
            with open(os.path.join(path, file), "r") as f:
                data = f.read()

            # Apply map-reduce summarization
            summary = map_reduce_summarize(data)

            # Calculate ROUGE score
            logging.info(f"Calculating ROUGE scores for {file_name}")
            rouge_score = {}
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(gold_summary, summary)
            rouge_score["rouge1"] = scores["rouge1"].fmeasure
            rouge_score["rouge2"] = scores["rouge2"].fmeasure
            rouge_score["rougeL"] = scores["rougeL"].fmeasure
            logging.info(f"ROUGE scores for {file_name}: {rouge_score}")
            
            id_score_map[file_name] = {
                "gold_summary": gold_summary,
                "generated_summary": summary,
                "rouge_score": rouge_score,
            }

            # Save the results to a JSON file after each document
            logging.info(f"Saving results for {file_name}")
            with open("./jaehyeok/results/summarization_map_reduce.json", "w") as f:
                json.dump(id_score_map, f, indent=4)
            
            logging.info(f"Successfully processed and saved results for {file_name}")
            
        except Exception as e:
            logging.error(f"Error processing document {file_name}: {str(e)}")
            logging.error("Continuing with next document...")
            continue
    
    logging.info("Map-reduce summarization process completed")

if __name__ == "__main__":
    main()
