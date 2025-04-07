import json
import os
import nltk
from nltk.tokenize import sent_tokenize
import textwrap
from typing import List, Dict

from ollama import ChatResponse, chat
from rouge_score import rouge_scorer
from tqdm import tqdm

# Download NLTK data for sentence tokenization if not already present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def chunk_text(text: str, max_chunk_size: int = 10000) -> List[str]:
    """Split text into chunks of approximately max_chunk_size characters."""
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
    
    return chunks

def adjust_chunk_size(text: str, initial_chunk_size: int = 10000) -> int:
    """Adjust chunk size to ensure the number of chunks is not too large."""
    # First try with initial chunk size
    chunks = chunk_text(text, initial_chunk_size)
    
    # If number of chunks is more than 10, adjust chunk size
    if len(chunks) > 10:
        # Calculate new chunk size to target 10 chunks
        total_length = sum(len(chunk) for chunk in chunks)
        new_chunk_size = total_length // 10
        return new_chunk_size
    
    return initial_chunk_size

def summarize_chunk(chunk: str, model: str = "qwen2.5:14b") -> str:
    """Summarize a single chunk of text using the specified model."""
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
    return response.message.content

def combine_summaries(summaries: List[str], model: str = "qwen2.5:14b") -> str:
    """Combine multiple chunk summaries into a final coherent summary."""
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
    return response.message.content

def map_reduce_summarize(text: str, model: str = "qwen2.5:14b", initial_chunk_size: int = 10000) -> str:
    """Apply map-reduce paradigm to summarize long text."""
    # Adjust chunk size if needed
    chunk_size = adjust_chunk_size(text, initial_chunk_size)
    
    # Map phase: split and summarize chunks
    chunks = chunk_text(text, chunk_size)
    print(f"Text split into {len(chunks)} chunks (chunk size: {chunk_size})")
    
    chunk_summaries = []
    for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
        summary = summarize_chunk(chunk, model)
        chunk_summaries.append(summary)
    
    # Reduce phase: combine summaries
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    
    print("Combining chunk summaries into final summary")
    final_summary = combine_summaries(chunk_summaries, model)
    return final_summary

def main():
    path = "./papers/postprocessed/full_texts/"
    files = os.listdir(path)

    gold_summaries = json.load(open("./abstractive_summaries/id_summary_map.json", "r"))
    
    # Try to load existing results, create new file if doesn't exist
    try:
        id_score_map = json.load(open("./experiments/summarization_map_reduce.json", "r"))
    except (FileNotFoundError, json.JSONDecodeError):
        id_score_map = {}

    for file in tqdm(files, desc="Processing files"):
        file_name = file.split(".")[0]

        if file_name in id_score_map:
            continue

        gold_summary = gold_summaries.get(file_name, None)
        if gold_summary is None:
            print(f"No gold summary found for {file_name}. Skipping...")
            continue

        with open(os.path.join(path, file), "r") as f:
            data = f.read()

        # Apply map-reduce summarization
        summary = map_reduce_summarize(data)

        # Calculate ROUGE score
        rouge_score = {}
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(gold_summary, summary)
        rouge_score["rouge1"] = scores["rouge1"].fmeasure
        rouge_score["rouge2"] = scores["rouge2"].fmeasure
        rouge_score["rougeL"] = scores["rougeL"].fmeasure
        
        id_score_map[file_name] = {
            "gold_summary": gold_summary,
            "generated_summary": summary,
            "rouge_score": rouge_score,
        }

        # Save the results to a JSON file
        with open("./experiments/summarization_map_reduce.json", "w") as f:
            json.dump(id_score_map, f, indent=4)

if __name__ == "__main__":
    main()
