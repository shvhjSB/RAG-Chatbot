# =========================
# FORCE HF CACHE TO D DRIVE (VERY IMPORTANT)
# =========================
import os
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["HF_HUB_CACHE"] = "D:/hf_cache"

# =========================
# IMPORTS
# =========================
import json
import torch
from dotenv import load_dotenv
from huggingface_hub import login

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    #BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# =========================
# LOAD ENV & LOGIN
# =========================
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF token missing in .env")

login(token=HF_TOKEN)

# =========================
# CONFIG
# =========================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "C:/Users/Admin/Desktop/robotic_project/dataset.jsonl"
OUTPUT_DIR = "models/tinyllama-lora"

# =========================
# LOAD DATASET (UNIVERSAL ZERO-SKIP JSONL LOADER)
# =========================
texts = []
skipped = 0

def flatten_json(obj):
    """Recursively collect all string values from JSON"""
    collected = []

    if isinstance(obj, dict):
        for v in obj.values():
            collected.extend(flatten_json(v))
    elif isinstance(obj, list):
        for item in obj:
            collected.extend(flatten_json(item))
    elif isinstance(obj, str):
        collected.append(obj)

    return collected


with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
            strings = flatten_json(obj)

            if strings:
                text = "\n".join(strings)
                texts.append(text)
            else:
                skipped += 1

        except Exception:
            skipped += 1

dataset = Dataset.from_dict({"text": texts})

print(f"✅ Loaded samples : {len(dataset)}")
print(f"⚠️ Skipped samples: {skipped}")

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

    # 🔥 THIS LINE FIXES EVERYTHING
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names
)

# =========================
# MODEL (4-bit, FAST & LIGHT)
# =========================
# from transformers import BitsAndBytesConfig

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True
# )
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu"
)



model = prepare_model_for_kbit_training(model)

model.gradient_checkpointing_disable()
model.config.use_cache = False


# =========================
# 🔥 BEST LoRA CONFIG FOR TinyLlama
# =========================
lora_config = LoraConfig(
    r=8,                     # smaller rank = faster & stable
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# TRAINING ARGS (OPTIMIZED)
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=500,
    fp16=False,
    logging_steps=50,
    report_to="none",
    optim="adamw_torch",
    dataloader_num_workers=0,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)




# =========================
# TRAIN
# =========================
trainer.train()

# =========================
# SAVE
# =========================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ TinyLlama LoRA training complete")
