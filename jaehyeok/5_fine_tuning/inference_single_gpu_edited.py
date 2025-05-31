import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model, _tok = None, None

def _load_model(model_dir: str) -> None:
    """한 번만 모델‧토크나이저를 로드해 전역 변수에 저장."""
    global _model, _tok
    if _model is not None:
        return

    _tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if _tok.eos_token_id is None:
        _tok.eos_token_id = _tok.convert_tokens_to_ids("</s>")

    base = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
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

    prompt = (
        "Summarize the following scientific paper:\n\n"
        f"{input_str}\n\nSummary:"
    )
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


if __name__ == "__main__":
    input_str = """
        input
        """
    
    output_str = get_answer(input_str)
    
    print("\n=== SUMMARY ===\n")
    print(output_str)