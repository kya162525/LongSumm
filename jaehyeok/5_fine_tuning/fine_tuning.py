import json, sys, os, logging, argparse
import random
from datetime import datetime
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from transformers import BitsAndBytesConfig
import torch.distributed as dist

# Hugging Face imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import wandb

# Add PEFT imports for LoRA
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)

# Create logs directory if it doesn't exist
os.makedirs("./jaehyeok/logs", exist_ok=True)
os.makedirs("./jaehyeok/models", exist_ok=True)

# Configure logging with timestamp in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"./jaehyeok/logs/fine_tuning_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Model configuration
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_INPUT_LENGTH = 32768
MAX_OUTPUT_LENGTH = 32768
TRAIN_RATIO = 0.95
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 3
SAVE_STEPS = 10
PUSH_TO_HUB_STEPS = 10

# LoRA configuration
USE_LORA = True  # Flag to enable/disable LoRA
LORA_R = 8  # LoRA attention dimension
LORA_ALPHA = 32  # Alpha parameter for LoRA
LORA_DROPOUT = 0.05  # Dropout probability for LoRA layers
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Modules to apply LoRA to

# Dataset filtering parameters
MAX_PAPER_LENGTH = 4*MAX_INPUT_LENGTH  # Maximum number of words in paper
MAX_SUMMARY_LENGTH = 4*MAX_OUTPUT_LENGTH  # Maximum number of words in summary

# Hugging Face Hub credentials
HUB_USERNAME = "JaehyeokLee"
HUB_TOKEN = "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"

def load_data() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load paper full texts and gold summaries with memory efficiency."""
    logging.info("Loading data...")
    
    # Load paper file paths rather than full contents
    path = "./papers/postprocessed/full_texts/"
    paper_file_paths = {}
    files = os.listdir(path)
    logging.info(f"Found {len(files)} paper files")
    
    for file in files:
        file_id = file.split(".")[0]
        paper_file_paths[file_id] = os.path.join(path, file)
            
    # Load gold summaries
    logging.info("Loading gold summaries")
    gold_summaries = json.load(open("./abstractive_summaries/id_summary_map.json", "r"))
    logging.info(f"Found {len(gold_summaries)} gold summaries")
    
    return paper_file_paths, gold_summaries

def prepare_dataset(paper_file_paths: Dict[str, str], gold_summaries: Dict[str, str]) -> Dataset:
    """Prepare dataset for fine-tuning with filtering conditions."""
    logging.info("Preparing dataset...")
    
    # Load valid IDs if available
    valid_ids = []
    if os.path.exists("./valid_ids.json"):
        with open("./valid_ids.json", 'r') as f:
            valid_ids = json.load(f)
        logging.info(f"Loaded {len(valid_ids)} valid paper IDs for filtering")
    
    # Track filtering statistics
    filtering_stats = {
        "total_papers": len(paper_file_paths),
        "no_gold_summary": 0,
        "not_in_valid_ids": 0,
        "paper_too_long": 0,
        "summary_too_long": 0,
        "accepted": 0
    }
    
    # Match papers with their summaries with filtering
    data = []
    for paper_id, paper_path in tqdm(paper_file_paths.items(), desc="Filtering papers"):
        # Read the paper text from file
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logging.warning(f"Error reading paper {paper_id}: {e}")
            continue
        # Skip if not in valid IDs list (if available and not empty)
        if valid_ids and paper_id not in valid_ids:
            # logging.info(f"Skipping {paper_id} - not in valid IDs")
            filtering_stats["not_in_valid_ids"] += 1
            continue
        
        # Skip if no gold summary available
        gold_summary = gold_summaries.get(paper_id, None)
        if gold_summary is None:
            logging.info(f"Skipping {paper_id} - no gold summary found")
            filtering_stats["no_gold_summary"] += 1
            continue
        
        # Check paper length
        paper_length = len(paper_text)
        if paper_length > MAX_PAPER_LENGTH:
            logging.info(f"Skipping {paper_id} - paper too long ({paper_length} characters)")
            filtering_stats["paper_too_long"] += 1
            continue
            
        # Check summary length
        summary_length = len(gold_summary)
        if summary_length > MAX_SUMMARY_LENGTH:
            logging.info(f"Skipping {paper_id} - summary too long ({summary_length} characters)")
            filtering_stats["summary_too_long"] += 1
            continue
        
        # If we get here, the paper passed all filters
        filtering_stats["accepted"] += 1
        data.append({
            "paper_id": paper_id,
            "text": paper_text,
            "summary": gold_summary
        })
    
    # Log filtering statistics
    logging.info("Filtering statistics:")
    for key, value in filtering_stats.items():
        logging.info(f"  {key}: {value}")
    
    logging.info(f"Created dataset with {len(data)} paper-summary pairs after filtering")
    return Dataset.from_list(data)

def split_dataset(dataset: Dataset) -> Tuple[Dataset, Dataset]:
    """Split dataset into training and validation sets."""
    logging.info(f"Splitting dataset: {TRAIN_RATIO*100}% train, {(1-TRAIN_RATIO)*100}% validation")
    train_val_split = dataset.train_test_split(test_size=(1-TRAIN_RATIO), seed=42)
    return train_val_split["train"], train_val_split["test"]

def preprocess_function(examples, tokenizer):
    prompts = [f"Summarize the following scientific paper:\n\n{text}\n\nSummary:" for text in examples["text"]]
    targets = examples["summary"]
    inputs, labels = [], []
    
    for prompt, target in zip(prompts, targets):
        # 프롬프트 토큰화
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        # 타겟 토큰화 (요약문)
        target_tokens = tokenizer(target, add_special_tokens=False)
        
        # 입력: 프롬프트 + 요약
        input_ids = prompt_tokens["input_ids"] + target_tokens["input_ids"] + [tokenizer.eos_token_id]
        # 레이블: 프롬프트 부분은 -100(loss 계산 제외), 요약 부분만 loss 계산
        label_ids = [-100] * len(prompt_tokens["input_ids"]) + target_tokens["input_ids"] + [tokenizer.eos_token_id]
        
        # 길이 제한
        if len(input_ids) > MAX_INPUT_LENGTH:
            input_ids = input_ids[:MAX_INPUT_LENGTH]
            label_ids = label_ids[:MAX_INPUT_LENGTH]
        
        # 패딩
        attention_mask = [1] * len(input_ids)
        padding_length = MAX_INPUT_LENGTH - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            label_ids = label_ids + [-100] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            
        inputs.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids
        })
    
    # 배치 형태로 변환
    batch = {
        "input_ids": torch.tensor([x["input_ids"] for x in inputs]),
        "attention_mask": torch.tensor([x["attention_mask"] for x in inputs]),
        "labels": torch.tensor([x["labels"] for x in inputs])
    }
    
    return batch


def apply_lora(model):
    """Apply LoRA adapters to the model."""
    logging.info("Applying LoRA adapters to the model")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Prepare model for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Log trainable parameters
    model.print_trainable_parameters()
    
    return model

def setup_model_and_tokenizer(local_rank: int):
    logging.info(f"Loading model/tokenizer on rank {local_rank}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank}, 
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_LORA:
        model = apply_lora(model)
    return model, tokenizer

def setup_training(model, tokenizer, train_dataset, val_dataset, run_name, local_rank):
    """Set up the training configuration."""
    logging.info("Setting up training configuration")
    
    # Define training arguments
    output_dir = f"./jaehyeok/models/{run_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        # eval_strategy="steps",
        eval_strategy="no",
        eval_steps=SAVE_STEPS,
        save_steps=SAVE_STEPS,
        # load_best_model_at_end=True,
        load_best_model_at_end=False,
        bf16=True,
        deepspeed="./jaehyeok/5_fine_tuning/ds_config.json",
        ddp_find_unused_parameters=False,      # (권장) LoRA 시 불필요 파라미터 탐지 끔
        report_to="wandb",
        logging_steps=1,
        log_level="info",
    )
    if local_rank != -1:
        training_args.local_rank = local_rank
    
    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # for causal LM like GPT, Qwen
    )
    
    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    return trainer

def push_to_hub(trainer, model_name, step):
    """Push the model to Hugging Face Hub at specific steps."""
    # Only push to hub from the main process (rank 0)
    if trainer.args.local_rank != 0:
        logging.info(f"Skipping push to hub on non-zero rank (rank={trainer.args.local_rank})")
        return
        
    logging.info(f"Pushing model to HuggingFace Hub at step {step}")
    save_dir = f"./jaehyeok/models/{model_name}-step-{step}"
    trainer.save_model(save_dir)
    try:
        if USE_LORA:
            # For LoRA models, we need to use PeftModel.from_pretrained
            model = PeftModel.from_pretrained(save_dir,
                    model_id=MODEL_NAME,
                    load_in_4bit=True,
                    torch_dtype="auto",
                    device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(save_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(save_dir)
            tokenizer = AutoTokenizer.from_pretrained(save_dir)
            
        repo_id = f"{HUB_USERNAME}/{model_name}"
        
        model.push_to_hub(repo_id,
            commit_message=f"Step {step} - LoRA model",
            token=HUB_TOKEN,
            exist_ok=True)
        tokenizer.push_to_hub(repo_id,
            commit_message=f"Step {step} - LoRA tokenizer",
            token=HUB_TOKEN,
            exist_ok=True)
        logging.info(f"Successfully pushed model to Hub as {model_name}-step-{step}")
    except Exception as e:
        logging.error(f"Failed to push model to Hub: {e}")

class PushToHubCallback(TrainerCallback):
    def __init__(self, trainer, model_name, push_every_n_steps=100):
        self.trainer = trainer
        self.model_name = model_name
        self.push_every_n_steps = push_every_n_steps
        self.best_metric = float('inf')

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.push_every_n_steps == 0:
            logging.info(f"Pushing model at step {state.global_step} to Hub: {self.model_name}")
            
            # 임시 저장 디렉토리 (반드시 필요한 경우에만 생성)
            temp_save_dir = f"./jaehyeok/temp_model"
            os.makedirs(temp_save_dir, exist_ok=True)
            
            try:
                # 모델과 토크나이저 저장
                kwargs['model'].save_pretrained(temp_save_dir)
                kwargs['tokenizer'].save_pretrained(temp_save_dir)
                
                # 허브에 직접 업로드
                if USE_LORA:
                    # LoRA 모델의 경우 어댑터만 업로드하여 공간 절약
                    model_repo_name = f"{self.model_name}-step-{state.global_step}-lora-{kwargs['model'].lora_rank}"
                else:
                    # 전체 모델 업로드
                    model_repo_name = f"{self.model_name}-step-{state.global_step}"
                kwargs['model'].push_to_hub(model_repo_name)
                kwargs['tokenizer'].push_to_hub(model_repo_name+"-tokenizer")
                logging.info(f"Model pushed to Hub as {model_repo_name}")
                
                # 임시 저장 디렉토리 정리
                self._clean_temp_dir(temp_save_dir)
                
            except Exception as e:
                logging.error(f"Failed to push model to Hub: {e}")
                self._clean_temp_dir(temp_save_dir)
    
    def _clean_temp_dir(self, directory):
        """임시 디렉토리 정리"""
        try:
            import shutil
            if os.path.exists(directory):
                shutil.rmtree(directory)
                logging.info(f"Cleaned temporary directory: {directory}")
        except Exception as e:
            logging.warning(f"Failed to clean temporary directory: {e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            if metrics["eval_loss"] < self.best_metric:
                self.best_metric = metrics["eval_loss"]
                logging.info(f"New best eval_loss {self.best_metric:.4f} at step {state.global_step}. Pushing best model.")
                push_to_hub(self.trainer, f"{self.model_name}-best", state.global_step)

class LogTrainLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            logging.info(f"[Step {step}] Training loss: {loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    print("sys.argv:", sys.argv)
    print(f"local_rank: {args.local_rank}")

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        
        # 초기화 후 world size와 rank 로깅
        if args.local_rank == 0:
            print(f"Initialized DDP: world_size={dist.get_world_size()}, local_rank={args.local_rank}")
    
    logging.info("Starting fine-tuning process")
    
    # Initialize wandb
    run_name = f"qwen3-8b-fine-tune-{timestamp}"
    if USE_LORA:
        run_name += "-lora"
        logging.info("LoRA is enabled with the following configuration:")
    
    # Load data
    paper_file_paths, gold_summaries = load_data()
    dataset = prepare_dataset(paper_file_paths, gold_summaries)
    
    # Print random samples to verify dataset content
    if args.local_rank <= 0:  # Only print on main process or single process mode
        logging.info("Verifying dataset content - random samples:")
        sample_indices = random.sample(range(len(dataset)), min(3, len(dataset)))
        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            logging.info(f"Sample {i+1}:")
            logging.info(f"Paper ID: {sample['paper_id']}")
            logging.info(f"Paper text (first 30 chars): {sample['text'][:30]}...")
            logging.info(f"Summary (first 30 chars): {sample['summary'][:30]}...")
            logging.info("-" * 40)
    wandb.init(project="qwen-summarization", name=run_name)
    
    # Split dataset
    train_dataset, val_dataset = split_dataset(dataset)
    logging.info(f"Training set: {len(train_dataset)} examples")
    logging.info(f"Validation set: {len(val_dataset)} examples")

    # 각 프로세스가 자기 GPU만 보도록
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    # rank 0 외에는 wandb 끔
    if args.local_rank not in (0, -1):
        os.environ["WANDB_MODE"] = "disabled"
    model, tokenizer = setup_model_and_tokenizer(local_rank=args.local_rank)
    
    # Preprocess datasets
    logging.info("Tokenizing datasets")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["paper_id", "text", "summary"],
        desc="Tokenizing training set"
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["paper_id", "text", "summary"],
        desc="Tokenizing validation set"
    )
    
    # Setup training
    trainer = setup_training(model, tokenizer, tokenized_train, tokenized_val, run_name, args.local_rank)
    trainer.add_callback(PushToHubCallback(trainer, f"jaehyeoklee/qwen3-8b-longsumm"))
    trainer.add_callback(LogTrainLossCallback())
    
    # Start training
    logging.info("Starting training")
    trainer.train()
    
    # Save and push final model
    if args.local_rank == 0:
        logging.info("Training completed, saving final model")
        trainer.save_model(f"./jaehyeok/models/{run_name}-final")
        push_to_hub(trainer, f"jaehyeoklee/qwen3-8b-longsumm-final", "final")
    
    # Finish wandb session
    if args.local_rank == 0:
        wandb.log({"final_model": f"jaehyeoklee/qwen3-8b-longsumm-final"})
        wandb.finish()
    logging.info("Fine-tuning process completed")

if __name__ == "__main__":
    main()
