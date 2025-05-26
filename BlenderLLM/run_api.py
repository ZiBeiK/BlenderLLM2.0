import os
import sys
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="",
)

append_text = ("这条指令是给出的物体非常简略，没有具体的描述。"
            "Let's create a bird feeder. Start with a cylindrical container that has multiple feeding ports around the sides. Add a perch beneath each feeding port. The top should have a lid that can be removed for refilling. Include a hook at the top for hanging the feeder."
            "Design a coffee table with a rectangular surface. Include a lower shelf for additional storage. Ensure the height is suitable for reaching from a sofa."
            "Create a router with a rectangular body and two antennas protruding from the back."
            "这3条指令对物体的描述比较详细，而且方便根据这些描述进行CAD scripts的生成。"
            "我希望可以把上面那条简略的指令按这几条详细指令的样子稍微扩写，不用非常复杂，只要写出这个物体的样子，比如各个构成部分的形状、颜色等（也不用都包含），方便我后续生成CAD就行了。"
            "帮我生成英文的指令，以Please draw开头，字数大概在20-40词左右。我只需要你说出这条详细的英文指令，不要有任何其他的语句，也不要告诉我你的回答有多少字，因为我希望能够直接把你返回给我的回答作为一个程序的输入。")

user_input = sys.argv[1]
user_input = append_text + "\nUser input: " + user_input + "\n"

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input},
    ],
)

prompt_text = completion.choices[0].message.content.strip()
print(prompt_text)

srun_command = (
    f'python chat.py --model_name ../models/BlenderLLM --prompt "{prompt_text}"'
)

os.system(srun_command)