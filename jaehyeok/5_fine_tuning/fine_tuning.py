import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm
from nltk.tokenize import word_tokenize

# Hugging Face imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
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
TRAIN_RATIO = 0.8
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
WARMUP_STEPS = 10
SAVE_STEPS = 100
PUSH_TO_HUB_STEPS = 100

# LoRA configuration
USE_LORA = True  # Flag to enable/disable LoRA
LORA_R = 8  # LoRA attention dimension
LORA_ALPHA = 32  # Alpha parameter for LoRA
LORA_DROPOUT = 0.05  # Dropout probability for LoRA layers
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Modules to apply LoRA to

# Dataset filtering parameters
MAX_PAPER_LENGTH = 4*MAX_INPUT_LENGTH  # Maximum number of words in paper
MAX_SUMMARY_LENGTH = 4*MAX_OUTPUT_LENGTH  # Maximum number of words in summary

def count_words(text: str) -> int:
    """Count the number of words in a text."""
    return len(word_tokenize(text))

def load_data() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load paper full texts and gold summaries."""
    logging.info("Loading data...")
    
    # Load paper full texts
    path = "./papers/postprocessed/full_texts/"
    paper_texts = {}
    files = os.listdir(path)
    logging.info(f"Found {len(files)} paper files")
    
    for file in tqdm(files, desc="Loading paper texts"):
        file_id = file.split(".")[0]
        with open(os.path.join(path, file), "r") as f:
            paper_texts[file_id] = f.read()
            
    # Load gold summaries
    logging.info("Loading gold summaries")
    gold_summaries = json.load(open("./abstractive_summaries/id_summary_map.json", "r"))
    logging.info(f"Found {len(gold_summaries)} gold summaries")
    
    return paper_texts, gold_summaries

def prepare_dataset(paper_texts: Dict[str, str], gold_summaries: Dict[str, str]) -> Dataset:
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
        "total_papers": len(paper_texts),
        "no_gold_summary": 0,
        "not_in_valid_ids": 0,
        "paper_too_long": 0,
        "summary_too_long": 0,
        "accepted": 0
    }
    
    # Match papers with their summaries with filtering
    data = []
    for paper_id, paper_text in tqdm(paper_texts.items(), desc="Filtering papers"):
        # Skip if not in valid IDs list (if available and not empty)
        if valid_ids and paper_id not in valid_ids:
            logging.info(f"Skipping {paper_id} - not in valid IDs")
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
        
        # # Keep track of lengths for the accepted papers
        # paper_word_count = count_words(paper_text)
        # summary_word_count = count_words(gold_summary)
        
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
    """Preprocess the examples by tokenizing inputs and targets."""
    inputs = [f"Summarize the following scientific paper:\n\n{text}" for text in examples["text"]]
    targets = examples["summary"]
    
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=MAX_OUTPUT_LENGTH, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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

def setup_model_and_tokenizer():
    """Load the model and tokenizer from Hugging Face, optionally applying LoRA."""
    logging.info(f"Loading model and tokenizer: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure tokenizer for generation if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA if enabled
    if USE_LORA:
        model = apply_lora(model)
        logging.info("LoRA adapters applied successfully")
    
    return model, tokenizer

def setup_training(model, tokenizer, train_dataset, val_dataset, run_name):
    """Set up the training configuration."""
    logging.info("Setting up training configuration")
    
    # Define training arguments
    output_dir = f"./jaehyeok/models/{run_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=10,
        remove_unused_columns=False,
        push_to_hub=False,
        # fp16=True
        bf16=True,
    )
    
    # Initialize data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=8,
    )
    
    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        # tokenizer=tokenizer,
        # label_names=["labels"],
    )
    
    return trainer

def push_to_hub(trainer, model_name, step):
    """Push the model to Hugging Face Hub at specific steps."""
    logging.info(f"Pushing model to HuggingFace Hub at step {step}")
    save_dir = f"./jaehyeok/models/{model_name}-step-{step}"
    trainer.save_model(save_dir)
    try:
        if USE_LORA:
            # For LoRA models, we need to use PeftModel.from_pretrained
            model = PeftModel.from_pretrained(save_dir,
                    load_in_4bit=True,           # 4-bit QLoRA
                    torch_dtype="auto",
                    device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(save_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(save_dir)
            tokenizer = AutoTokenizer.from_pretrained(save_dir)
            
        model.push_to_hub(f"{model_name}-step-{step}")
        tokenizer.push_to_hub(f"{model_name}-step-{step}")
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
            push_to_hub(self.trainer, self.model_name, state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            if metrics["eval_loss"] < self.best_metric:
                self.best_metric = metrics["eval_loss"]
                logging.info(f"New best eval_loss {self.best_metric:.4f} at step {state.global_step}. Pushing best model.")
                push_to_hub(self.trainer, f"{self.model_name}-best", state.global_step)

def main():
    logging.info("Starting fine-tuning process")
    
    # Initialize wandb
    run_name = f"qwen3-8b-fine-tune-{timestamp}"
    if USE_LORA:
        run_name += "-lora"
    wandb.init(project="qwen-summarization", name=run_name)
    
    # Log LoRA configuration if enabled
    if USE_LORA:
        logging.info("LoRA is enabled with the following configuration:")
        logging.info(f"  LoRA rank (r): {LORA_R}")
        logging.info(f"  LoRA alpha: {LORA_ALPHA}")
        logging.info(f"  LoRA dropout: {LORA_DROPOUT}")
        logging.info(f"  LoRA target modules: {LORA_TARGET_MODULES}")
    
    # Load data
    paper_texts, gold_summaries = load_data()
    
    # Prepare dataset
    dataset = prepare_dataset(paper_texts, gold_summaries)
    
    # Split dataset
    train_dataset, val_dataset = split_dataset(dataset)
    logging.info(f"Training set: {len(train_dataset)} examples")
    logging.info(f"Validation set: {len(val_dataset)} examples")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
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
    trainer = setup_training(model, tokenizer, tokenized_train, tokenized_val, run_name)
    
    # Add callback for pushing to Hub
    push_callback = PushToHubCallback(trainer, f"jaehyeoklee/qwen3-8b-longsumm")
    trainer.add_callback(push_callback)
    
    # Start training
    logging.info("Starting training")
    trainer.train()
    
    # Save and push final model
    logging.info("Training completed, saving final model")
    trainer.save_model(f"./jaehyeok/models/{run_name}-final")
    push_to_hub(trainer, f"jaehyeoklee/qwen3-8b-longsumm-final", "final")
    
    # Finish wandb session
    wandb.finish()
    logging.info("Fine-tuning process completed")

if __name__ == "__main__":
    main()
