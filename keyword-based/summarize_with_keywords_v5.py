import json
import os

from ollama import ChatResponse, chat
from rouge_score import rouge_scorer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

full_text_path = "./papers/postprocessed/full_texts/"
json_path = "./papers/postprocessed/jsons/"
full_text_files = os.listdir(full_text_path)

valid_ids = json.load(open("./valid_ids.json", "r"))
gold_summaries = json.load(open("./abstractive_summaries/id_summary_map.json", "r"))
if not os.path.exists("./experiments/summarization_with_keywords_v5.json"):
    id_score_map = {}
else:
    id_score_map = json.load(open("./experiments/summarization_with_keywords_v5.json", "r"))

model = SentenceTransformer("all-MiniLM-L6-v2")

for file in tqdm(full_text_files):
    file_name = file.split(".")[0]

    if file_name not in valid_ids or file_name in id_score_map:
        continue

    gold_summary = gold_summaries.get(file_name, None)
    if gold_summary is None:
        print(f"No gold summary found for {file_name}. Skipping...")
        continue

    with open(os.path.join(full_text_path, file), "r") as f:
        data = f.read()

    json_data = json.load(open(os.path.join(json_path, file_name + ".json"), "r"))
    source_text = ""
    for section in json_data["sections"]:
        if "intro" in section["heading"].lower():
            source_text = section["text"]
            break

    if source_text == "":  # If no intro section is found, use the abstract text
        source_text = json_data["abstractText"]

    # Step 1: Extract keywords using qwen2.5 model
    keyword_response: ChatResponse = chat(
        model="qwen2.5:14b",
        messages=[
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are a helpful assistant who extracts keywords from scientific documents.",
                        "Please extract 20 distinctive keywords from the text, which can represent the main topics and contributions of the paper.",
                        "Your response should be a comma-separated list of keywords.",
                        "Do not include any other text or explanations.",
                    ]
                ),
            },
            {
                "role": "user",
                "content": "Can you extract 20 keywords from the following text: ",
            },
            {
                "role": "user",
                "content": source_text,
            },
        ],
    )
    print("Keyword Response:")
    print(keyword_response.message.content)
    keywords = keyword_response.message.content.split(",")
    print("Extracted Keywords:")
    print(keywords)

    # Step 2: Extract relevant chunks for each keyword using Sentence-BERT and cosine similarity with semantic chunking

    # Split the document into sentences
    sentences = data.split(". ")
    sentence_embeddings = model.encode(sentences)

    # Perform semantic chunking by combining consecutive sentences with high similarity
    threshold = 0.4  # Define a similarity threshold for chunking
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity([sentence_embeddings[i - 1]], [sentence_embeddings[i]])[0][0]
        if similarity > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    chunk_idx_map = {}
    for i, chunk in enumerate(chunks):
        chunk_idx_map[chunk] = i

    # Embed the chunks
    chunk_embeddings = model.encode(chunks)

    # Embed the keywords
    keyword_embeddings = model.encode(keywords)

    # Find top-10 relevant chunks for each keyword
    relevant_chunks = set()
    for keyword, keyword_embedding in zip(keywords, keyword_embeddings):
        similarities = cosine_similarity([keyword_embedding], chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-20:][::-1]  # Get indices of top-20 chunks
        for idx in top_indices:
            relevant_chunks.add(chunks[idx])  # Add the chunk to the set to deduplicate

    # Combine relevant chunks into a single text
    chunk_list = list(relevant_chunks)
    print("Number of relevant chunks:", len(chunk_list))

    # Sort the chunks based on their original order in the document
    chunk_list.sort(key=lambda x: chunk_idx_map[x])

    # Step 3: Summarize the deduplicated sentences iteratively
    deduplicated_text = "=====\n".join(chunk_list)

    summary_response: ChatResponse = chat(
        model="qwen2.5:14b",
        messages=[
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are a helpful assistant who organizes scientific content.",
                        "Please concatenate the provided sentences into a coherent and natural-flowing paragraph.",
                        "Do not summarize or omit any information from the sentences.",
                        "Ensure the paragraph maintains the original meaning and structure of the provided sentences.",
                        "Focus on preserving the content and logical flow of the text.",
                    ]
                ),
            },
            {"role": "user", "content": "Can you organize the following sentences into a coherent paragraph: "},
            {
                "role": "user",
                "content": deduplicated_text,
            },
        ],
    )
    summary = summary_response.message.content

    for iteration in range(3):
        print(f"Iteration {iteration + 1}:")

        # Enrich the summary by comparing it with the original text
        enrichment_response: ChatResponse = chat(
            model="qwen2.5:14b",
            messages=[
                {
                    "role": "system",
                    "content": "\n".join(
                        [
                            "You are a helpful assistant who enriches summaries by comparing them with the original text.",
                            "Please identify any missing information from the original text and incorporate it into the summary.",
                            "Ensure the enriched summary is coherent and logically structured.",
                        ]
                    ),
                },
                {"role": "user", "content": "Here is the original text: "},
                {"role": "user", "content": deduplicated_text},
                {"role": "user", "content": "Here is the current summary: "},
                {"role": "user", "content": summary},
            ],
        )
        summary = enrichment_response.message.content

    if "###" in summary:
        summary_response: ChatResponse = chat(
            model="qwen2.5:14b",
            messages=[
                {
                    "role": "system",
                    "content": "\n".join(
                        [
                            "You are a helpful assistant that converts text written in Markdown format into natural, fluent prose.",
                            "Your task is to rewrite the given content as continuous paragraph text without preserving the original Markdown structure, headings, bullet points, or formatting symbols.",
                            "Keep all important information, but present it in a coherent, readable form as if explaining it in an article or essay.",
                        ]
                    ),
                },
                {
                    "role": "user",
                    "content": "Can you rewrite the following markdown-style text into a continuous paragraph: ",
                },
                {
                    "role": "user",
                    "content": summary,
                },
            ],
        )
        summary = summary_response.message.content

    print("Generated Summary:")
    print(summary)

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
    with open("./experiments/summarization_with_keywords_v5.json", "w") as f:
        json.dump(id_score_map, f, indent=4)
