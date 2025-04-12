import json
import os
import numpy as np
from collections import defaultdict

def read_json_file(file_path):
    """
    Read and parse a JSON file
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Parsed JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file - {file_path}")
        return None

def write_json_file(data, file_path, indent=4):
    """
    Write data to a JSON file
    
    Args:
        data (dict): Data to write
        file_path (str): Path to save the JSON file
        indent (int): Number of spaces for indentation
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        print(f"Successfully wrote JSON to {file_path}")
    except Exception as e:
        print(f"Error writing JSON file: {e}")

def parse_json_string(json_string):
    """
    Parse a JSON string
    
    Args:
        json_string (str): JSON string to parse
        
    Returns:
        dict: Parsed JSON data
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        print("Error: Invalid JSON string format")
        return None

def analyze_summarization_results(file_path):
    """
    Analyze the summarization results JSON file and calculate statistics
    
    Args:
        file_path (str): Path to the JSON file containing summarization results
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file - {file_path}")
        return

    # Initialize statistics
    rouge_scores = defaultdict(list)
    gold_lengths = []
    generated_lengths = []
    length_ratios = []
    total_entries = len(data)
    
    # Collect all ROUGE scores and lengths
    for entry_id, entry_data in data.items():
        if 'rouge_score' in entry_data:
            for metric, score in entry_data['rouge_score'].items():
                rouge_scores[metric].append(score)
        
        # Collect summary lengths
        if 'gold_summary' in entry_data:
            gold_length = len(entry_data['gold_summary'].split())
            gold_lengths.append(gold_length)
        
        if 'generated_summary' in entry_data:
            generated_length = len(entry_data['generated_summary'].split())
            generated_lengths.append(generated_length)
            
            # Calculate length ratio if both summaries exist
            if gold_length > 0:
                length_ratio = generated_length / gold_length
                length_ratios.append(length_ratio)
    
    # Calculate statistics
    print("\n=== Summarization Results Analysis ===")
    print(f"Total number of entries: {total_entries}")
    
    # Print ROUGE score statistics
    print("\nROUGE Score Statistics:")
    for metric, scores in rouge_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {mean_score:.4f}")
        print(f"  Std Dev: {std_score:.4f}")
        print(f"  Min: {min_score:.4f}")
        print(f"  Max: {max_score:.4f}")
        
        # Calculate score distribution
        score_ranges = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
        print("\n  Score Distribution:")
        for lower, upper in score_ranges:
            count = sum(1 for score in scores if lower <= score < upper)
            percentage = (count / len(scores)) * 100
            print(f"    {lower:.1f}-{upper:.1f}: {count} entries ({percentage:.1f}%)")
    
    # Print length statistics
    print("\nSummary Length Statistics:")
    
    def print_length_stats(name, lengths):
        if not lengths:
            return
        
        q1 = np.percentile(lengths, 25)
        median = np.percentile(lengths, 50)
        q3 = np.percentile(lengths, 75)
        mean = np.mean(lengths)
        std = np.std(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        
        print(f"\n{name}:")
        print(f"  Mean: {mean:.2f} words")
        print(f"  Std Dev: {std:.2f} words")
        print(f"  Min: {min_len} words")
        print(f"  Max: {max_len} words")
        print(f"  Q1: {q1:.2f} words")
        print(f"  Median: {median:.2f} words")
        print(f"  Q3: {q3:.2f} words")
        
        # Print length distribution
        length_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, float('inf'))]
        print("\n  Length Distribution:")
        for lower, upper in length_ranges:
            count = sum(1 for length in lengths if lower <= length < upper)
            percentage = (count / len(lengths)) * 100
            if upper == float('inf'):
                print(f"    {lower}+ words: {count} entries ({percentage:.1f}%)")
            else:
                print(f"    {lower}-{upper} words: {count} entries ({percentage:.1f}%)")
    
    print_length_stats("Gold Summary", gold_lengths)
    print_length_stats("Generated Summary", generated_lengths)
    
    # Print length ratio statistics
    if length_ratios:
        print("\nLength Ratio Statistics (Generated/Gold):")
        q1 = np.percentile(length_ratios, 25)
        median = np.percentile(length_ratios, 50)
        q3 = np.percentile(length_ratios, 75)
        mean = np.mean(length_ratios)
        std = np.std(length_ratios)
        min_ratio = min(length_ratios)
        max_ratio = max(length_ratios)
        
        print(f"  Mean: {mean:.2f}")
        print(f"  Std Dev: {std:.2f}")
        print(f"  Min: {min_ratio:.2f}")
        print(f"  Max: {max_ratio:.2f}")
        print(f"  Q1: {q1:.2f}")
        print(f"  Median: {median:.2f}")
        print(f"  Q3: {q3:.2f}")
        
        # Print ratio distribution
        ratio_ranges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, float('inf'))]
        print("\n  Ratio Distribution:")
        for lower, upper in ratio_ranges:
            count = sum(1 for ratio in length_ratios if lower <= ratio < upper)
            percentage = (count / len(length_ratios)) * 100
            if upper == float('inf'):
                print(f"    {lower}+: {count} entries ({percentage:.1f}%)")
            else:
                print(f"    {lower}-{upper}: {count} entries ({percentage:.1f}%)")

def main():
    # Example usage
    json_file = "/home/LongSumm/jaehyeok/results/summarization_map_reduce.json"
    analyze_summarization_results(json_file)

if __name__ == "__main__":
    main()
