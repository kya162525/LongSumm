import os, json, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model, _tok = None, None

def _load_model(model_dir: str) -> None:
    """한 번만 모델‧토크나이저를 로드해 전역 변수에 저장."""
    global _model, _tok
    if _model is not None:
        return

    _tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, token=HF_TOKEN)
    if _tok.eos_token_id is None:
        _tok.eos_token_id = _tok.convert_tokens_to_ids("</s>")

    base = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, token=HF_TOKEN
    )
    try:
        base = PeftModel.from_pretrained(base, model_dir)
    except Exception:
        pass

    _model = base.to(_device).eval()

def get_answer(
    input_str: str,
    model_dir: str = "JaehyeokLee/qwen3-8b-longsumm-extractive-datasets-final",
    max_input_length: int = 30_000,
    max_output_length: int = 1_024,
    num_beams: int = 4,
) -> str:
    _load_model(model_dir)

    # prompt = (
    #     "Summarize the following scientific paper:\n\n"
    #     f"{input_str}\n\nSummary:"
    # )
    prompt = input_str.strip()
    inputs = _tok(
        prompt,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        gen = _model.generate(
            **inputs,
            max_new_tokens=max_output_length,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            early_stopping=True,
            eos_token_id=_tok.eos_token_id,
        )

    summary = _tok.decode(
        gen[0, inputs["input_ids"].shape[1]:],  # 프롬프트 이후 토큰만
        skip_special_tokens=True,
    )
    return summary.strip()


# if __name__ == "__main__":
#     input_str = """
#         hello
#         """
    
#     output_str = get_answer(input_str)
    
#     print("\n=== SUMMARY ===\n")
#     print(output_str)



if __name__ == "__main__":
    json_path = "./papers/test/json_files/"
    files = os.listdir(json_path)
    extractive_summaries = json.load(open("./extractive_summaries/test4_output.json"))

    input_str = """
<Scientific Document>: {{full_text}}
=========================
<Previous Summary>: {{summary}}
=========================
You are a helpful assistant who generates entity-dense summaries of the scientific document above.
Identify missing information from the previously generated summary and add it to the new summary.
Write a new, longer summary which covers every entity and detail from the previous summary plus the new information.
Remember to make the summary longer than the previous summary.
"""

    output = {}

    print(files)
    
    for file in tqdm(files):
        file_name = file.split(".")[0]
        summary = extractive_summaries[file_name]

        longest_summary = ""
        for i in range(3):
            print(f"Iteration {i + 1}...")
            prompt = input_str.replace("{{full_text}}", summary).replace("{{summary}}", summary)
            output_str = get_answer(prompt)
            print(f"Generated summary word count: {len(output_str.split())}")

            if len(output_str.split()) > len(longest_summary.split()):
                longest_summary = output_str

        print("Generated Summary:")
        print(longest_summary[:100])
        output[file_name] = longest_summary

    with open("./experiments/testing_ext_then_abs_ft.json", "w") as f:
        json.dump(output, f, indent=4)