from huggingface_hub import login, HfApi
from peft import PeftModel, PeftConfig
import os

# Load your HF token, either from an environment variable or user input
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    hf_token = "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"

# Login to Hugging Face
login(token=hf_token)

# Path to your adapter
adapter_path = "/root/workspace/LongSumm/jaehyeok/models/qwen3-8b-fine-tune-20250520_072828-lora"

# Define your HF repository name
repo_name = "JaehyeokLee/qwen3-8b-lora-summarization-checkpoints"  # Change to your username

# Create API object
api = HfApi()

# Create the repository if it doesn't exist
api.create_repo(
    repo_id=repo_name,
    repo_type="model",
    exist_ok=True  # Won't fail if the repo already exists
)

# Push the adapter to HF Hub
api.upload_folder(
    folder_path=adapter_path,
    repo_id=repo_name,
    repo_type="model",
)

print(f"Successfully pushed adapter to {repo_name}")