import json
import os

from ollama import ChatResponse, chat
from rouge_score import rouge_scorer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

path = "./papers/postprocessed/full_texts/"
files = os.listdir(path)

valid_ids = json.load(open("./valid_ids.json", "r"))
gold_summaries = json.load(open("./abstractive_summaries/id_summary_map.json", "r"))
id_score_map = json.load(open("./experiments/summarization_with_keywords.json", "r"))
model = SentenceTransformer("all-MiniLM-L6-v2")

for file in tqdm(files):
    file_name = file.split(".")[0]

    if file_name not in valid_ids or file_name in id_score_map:
        continue

    gold_summary = gold_summaries.get(file_name, None)
    if gold_summary is None:
        print(f"No gold summary found for {file_name}. Skipping...")
        continue

    with open(os.path.join(path, file), "r") as f:
        data = f.read()

    # Step 1: Extract keywords using qwen2.5 model
    keyword_response: ChatResponse = chat(
        model="qwen2.5:14b",
        messages=[
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are a helpful assistant who extracts keywords from scientific documents.",
                        "Please extract 10 distinctive and multi-perspective keywords from the text.",
                        "Your response should be a comma-separated list of keywords.",
                        "Do not include any other text or explanations.",
                    ]
                ),
            },
            {
                "role": "user",
                "content": "Can you extract 10 distinctive and multi-perspective keywords from the following text: ",
            },
            {
                "role": "user",
                "content": data[:1000],  # Limit to the first 1000 characters for keyword extraction
            },
        ],
    )
    keywords = keyword_response.message.content.split(",")[:10]  # Extract top 10 keywords
    print("Extracted Keywords:")
    print(keywords)

    # Step 2: Extract relevant sentences for each keyword using Sentence-BERT and cosine similarity

    # Split the document into sentences
    sentences = data.split(". ")
    sentence_embeddings = model.encode(sentences)

    # Embed the keywords
    keyword_embeddings = model.encode(keywords)

    # Find top-10 relevant sentences for each keyword
    relevant_sentences = set()
    for keyword, keyword_embedding in zip(keywords, keyword_embeddings):
        similarities = cosine_similarity([keyword_embedding], sentence_embeddings)[0]
        top_indices = np.argsort(similarities)[-10:][::-1]  # Get indices of top-10 sentences
        for idx in top_indices:
            relevant_sentences.add(sentences[idx])  # Add the sentence to the set to deduplicate

    # Step 3: Summarize the deduplicated sentences
    deduplicated_text = " ".join(relevant_sentences)
    summary_response: ChatResponse = chat(
        model="qwen2.5:14b",
        messages=[
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are a helpful assistant who summarizes scientific documents.",
                        "Please summarize the text provided to you.",
                        "Your summary should be concise and capture the main points of the text.",
                        "Do not structure your response in bullet points or lists, but rather in a coherent paragraph.",
                    ]
                ),
            },
            {"role": "user", "content": "Can you summarize the following text: "},
            {
                "role": "user",
                "content": deduplicated_text,
            },
        ],
    )
    summary = summary_response.message.content
    print("Generated Summary:")
    print(summary)
    print("Gold Summary:")
    print(gold_summary)

    # calculate ROUGE score
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
    print("ROUGE Score:")
    print(rouge_score)
    print("=============================")

    # Save the results to a JSON file
    with open("./experiments/summarization_with_keywords.json", "w") as f:
        json.dump(id_score_map, f, indent=4)
