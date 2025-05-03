import json
import os
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict
import statistics
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data for sentence tokenization if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def chunk_text(text: str, max_chunk_size: int = 4000) -> List[str]:
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

def analyze_chunks(chunks: List[str]) -> Dict:
    """Analyze statistics for a list of chunks."""
    chunk_sizes = [len(chunk) for chunk in chunks]
    
    return {
        "num_chunks": len(chunks),
        "avg_chunk_size": statistics.mean(chunk_sizes),
        "median_chunk_size": statistics.median(chunk_sizes),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes),
        "std_chunk_size": statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0,
        "chunk_sizes": chunk_sizes
    }

def main():
    # Define max chunk size
    max_chunk_size = 10000  # You can change this value as needed
    
    path = "./papers/postprocessed/full_texts/"
    files = os.listdir(path)
    
    # Initialize statistics
    all_stats = []
    chunk_counts = []
    
    for file in files:
        file_name = file.split(".")[0]
        
        with open(os.path.join(path, file), "r") as f:
            data = f.read()
        
        chunks = chunk_text(data, max_chunk_size)
        stats = analyze_chunks(chunks)
        stats["file_name"] = file_name
        
        all_stats.append(stats)
        chunk_counts.append(stats["num_chunks"])
    
    # Calculate overall statistics
    overall_stats = {
        "total_files": len(files),
        "avg_chunks_per_file": statistics.mean(chunk_counts),
        "median_chunks_per_file": statistics.median(chunk_counts),
        "min_chunks_per_file": min(chunk_counts),
        "max_chunks_per_file": max(chunk_counts),
        "std_chunks_per_file": statistics.stdev(chunk_counts) if len(chunk_counts) > 1 else 0,
        "chunk_count_distribution": dict(Counter(chunk_counts)),
        "max_chunk_size": max_chunk_size
    }
    
    # Save detailed statistics with max_chunk_size in filename
    output_filename = f"./experiments/chunk_analysis_{max_chunk_size}.json"
    with open(output_filename, "w") as f:
        json.dump({
            "overall_stats": overall_stats,
            "per_file_stats": all_stats
        }, f, indent=4)
    
    # Create visualization with max_chunk_size in filename
    plt.figure(figsize=(10, 6))
    sns.histplot(chunk_counts, bins=20)
    plt.title(f"Distribution of Number of Chunks per Document (Max Chunk Size: {max_chunk_size})")
    plt.xlabel("Number of Chunks")
    plt.ylabel("Number of Documents")
    plt.savefig(f"./experiments/chunk_distribution_{max_chunk_size}.png")
    plt.close()
    
    # Print summary statistics
    print("\nChunking Analysis Summary:")
    print(f"Max chunk size: {max_chunk_size}")
    print(f"Total files analyzed: {overall_stats['total_files']}")
    print(f"Average chunks per file: {overall_stats['avg_chunks_per_file']:.2f}")
    print(f"Median chunks per file: {overall_stats['median_chunks_per_file']}")
    print(f"Min chunks per file: {overall_stats['min_chunks_per_file']}")
    print(f"Max chunks per file: {overall_stats['max_chunks_per_file']}")
    print(f"Standard deviation of chunks per file: {overall_stats['std_chunks_per_file']:.2f}")
    
    print("\nChunk count distribution:")
    for count, freq in sorted(overall_stats['chunk_count_distribution'].items()):
        print(f"{count} chunks: {freq} documents")

if __name__ == "__main__":
    main() 