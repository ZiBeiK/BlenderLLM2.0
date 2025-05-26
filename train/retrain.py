import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from dataset.dataset import NameToInstructionWithCriteriaDataset

# 自动从模型目录加载 tokenizer（自动识别 tokenizer.json 和配置）
tokenizer = AutoTokenizer.from_pretrained(
    "/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    local_files_only=True
)

# 加载模型，并设置 bfloat16 精度
model = AutoModelForCausalLM.from_pretrained(
    "/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# 数据集加载
train_dataset = NameToInstructionWithCriteriaDataset(
    jsonl_path="dataset/CADBench_train.jsonl",
    tokenizer=tokenizer,
    max_length=512
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./checkpoints_cadbench",
    num_train_epochs=20,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,
    fp16=False,  # 不用 fp16
    bf16=torch.cuda.is_bf16_supported(), 
    remove_unused_columns=False,
    report_to="none",
    deepspeed='ds_config.json'
)

# Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

trainer.save_model("./final_model_cadbench")
tokenizer.save_pretrained("./final_model_cadbench")
