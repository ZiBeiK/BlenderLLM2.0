import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# 路径配置
# model_path = "/remote-home1/wxzhang/BlenderLLM/deepseek/final_model_cadbench"
model_path = "/remote-home1/wxzhang/BlenderLLM/deepseek/final_model"
input_file = "dataset/CADBench_test.jsonl"
output_file = "dataset/output/instructions_nocadbench.json"

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()

# 固定前缀模板
append_text = (
    "You are a helpful assistant for 3D modeling. Your job is to rewrite a vague object name "
    "into a highly detailed design instruction in English, suitable for generating CAD or 3D models.\n\n"
    "Please follow these rules:\n"
    "- Output only the instruction (no preamble or explanation).\n"
    "- The instruction must be structured and vivid.\n"
    "- You can mention some of the followings: the shape, structure, components, size, materials, and color.\n"
    "- Describe how parts are connected or arranged in space.\n"
    "- Be imaginative but concrete.\n"
    "- Around 100 to 150 words.\n\n"
    "User input: {name}\n"
    "Instruction:"
)

# 加载输入数据
data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # 忽略空行
            data.append(json.loads(line))
            
# 开始生成
results = []
for item in tqdm(data, desc="Generating Instructions"):
    name = item.get("name", "").strip()
    criteria = item.get("criteria", "")
    if not name:
        continue

    full_prompt = append_text + f"\nUser input: {name}\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    instruction = response[len(full_prompt):].strip()

    results.append({
        "name": name,
        "instruction": instruction,
        "criteria": criteria,
    })

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Instruction generation complete. Output saved to {output_file}")
