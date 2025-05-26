from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import sys

model_path = "../train/retrained_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()

append_text = (
    "You are a helpful assistant that rewrites vague 3D modeling instructions into clearer, detailed English ones. "
    "Given a simple prompt about an object, generate a detailed instruction suitable for CAD modeling. "
    "The instruction should:\n"
    "- Be in English\n"
    "- Start with: 'Please draw'\n"
    "- Contain 20 to 40 words\n"
    "- Avoid explanations, apologies, or any other commentary\n"
    "- Output only the instruction and nothing else.\n"
)

user_input = sys.argv[1]
full_prompt = append_text + "\nUser input: " + user_input + "\n"

inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id
)

prompt_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
prompt_text = prompt_text[len(full_prompt):].strip()
print(prompt_text)

srun_command = (
    f'python chat.py --model_name ../models/BlenderLLM --prompt "{prompt_text}"'
)

os.system(srun_command)