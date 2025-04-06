import json
import os

from ollama import ChatResponse, chat
from rouge_score import rouge_scorer
from tqdm import tqdm

path = "./papers/postprocessed/full_texts/"
files = os.listdir(path)

gold_summaries = json.load(open("./abstractive_summaries/id_summary_map.json", "r"))
id_score_map = {}

for file in tqdm(files):
    file_name = file.split(".")[0]
    gold_summary = gold_summaries.get(file_name, None)
    if gold_summary is None:
        print(f"No gold summary found for {file_name}. Skipping...")
        continue

    with open(os.path.join(path, file), "r") as f:
        data = f.read()

    response: ChatResponse = chat(
        model="qwen2.5:14b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who summarizes scientific documents. Please provide a concise summary of the text.",
            },
            {"role": "user", "content": "Can you summarize the following text: "},
            {
                "role": "user",
                "content": data,
            },
        ],
    )
    summary = response.message.content

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

    # Save the results to a JSON file
    with open("./experiments/summarization_with_full_text.json", "w") as f:
        json.dump(id_score_map, f, indent=4)
