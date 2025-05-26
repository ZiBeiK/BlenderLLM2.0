import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from dataset.dataset import NameToInstructionDataset

tokenizer = AutoTokenizer.from_pretrained(
    "../models/Qwen2.5-1.5B",
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    "../models/Qwen2.5-1.5B",
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

train_dataset = NameToInstructionDataset(
    jsonl_path="dataset/BlendNet.jsonl",
    tokenizer=tokenizer,
    max_length=256
)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(), 
    remove_unused_columns=False,
    report_to="none",
    deepspeed='ds_config.json'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
