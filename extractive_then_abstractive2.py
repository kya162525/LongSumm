import json
import os

from ollama import ChatResponse, chat
from rouge_score import rouge_scorer
from tqdm import tqdm

extractive_summaries = json.load(open("./extractive_summaries/train4_output.json"))
path = "./papers/postprocessed/full_texts/"
files = os.listdir(path)

valid_ids = json.load(open("./valid_ids.json", "r"))
gold_summaries = json.load(open("./abstractive_summaries/id_summary_map.json", "r"))
if not os.path.exists("./experiments/extractive_then_abstractive2.json"):
    id_score_map = {}
else:
    id_score_map = json.load(open("./experiments/extractive_then_abstractive2.json", "r"))

SYSTEM_PROMPT = """
<Scientific Document>: {{full_text}}
=========================
<Previous Summary>: {{summary}}
=========================
You are a helpful assistant who generates entity-dense summaries of the scientific document above.
Identify missing information from the previously generated summary and add it to the new summary.
Write a new, longer summary which covers every entity and detail from the previous summary plus the new information.
Remember to make the summary longer than the previous summary.
"""

for file in tqdm(files):
    paper_id = file.split(".")[0]
    if paper_id in id_score_map or paper_id not in valid_ids or paper_id not in extractive_summaries:
        continue

    gold_summary = gold_summaries[paper_id]
    extractive_summary = extractive_summaries[paper_id]
    if extractive_summary == "":
        continue

    with open(os.path.join(path, file), "r") as f:
        full_text = f.read()

    print(f"\nProcessing paper ID: {paper_id}")
    print(
        f"extractive summary word count: {len(extractive_summary.split())}, gold summary word count: {len(gold_summary.split())}"
    )

    summary = extractive_summary
    longest_summary = ""
    for i in range(3):
        print(f"Iteration {i + 1}...")
        prompt = SYSTEM_PROMPT.replace("{{full_text}}", extractive_summary).replace("{{summary}}", summary)
        response: ChatResponse = chat(
            model="qwen2.5:14b-8k",
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                }
            ],
        )
        summary = response.message.content
        print(f"Generated summary word count: {len(summary.split())}")

        if len(summary.split()) > len(longest_summary.split()):
            longest_summary = summary

    # calculate ROUGE score
    rouge_score = {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(gold_summary, longest_summary)
    rouge_score["rouge1"] = scores["rouge1"].fmeasure
    rouge_score["rouge2"] = scores["rouge2"].fmeasure
    rouge_score["rougeL"] = scores["rougeL"].fmeasure
    id_score_map[paper_id] = {
        "gold_summary": gold_summary,
        "generated_summary": summary,
        "rouge_score": rouge_score,
    }

    # Save the results to a JSON file
    with open("./experiments/extractive_then_abstractive2.json", "w") as f:
        json.dump(id_score_map, f, indent=4)
