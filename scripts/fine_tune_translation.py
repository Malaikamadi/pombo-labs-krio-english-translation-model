from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import torch

# 1. Load  dataset
dataset = load_dataset('json', data_files='krio_en_pairs_cleaned.jsonl', split='train')

# 2. Load tokenizer and model
model_checkpoint = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Set source and target language before tokenizing
tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "eng_Latn"

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_checkpoint,
    device_map="auto",
    torch_dtype="auto"  
)

# 3. Preprocessing function
def preprocess_function(examples):
    inputs = examples['krio']
    targets = examples['en']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 4. Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["krio", "en"])

# 5. Prepare data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 6. Set training arguments for CPU and low memory use
training_args = TrainingArguments(
    output_dir="./results",                
    learning_rate=2e-5,
    per_device_train_batch_size=2,         
    per_device_eval_batch_size=2,
    num_train_epochs=15,                   
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    no_cuda=True                           
)

# 7. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 8. training
trainer.train()
