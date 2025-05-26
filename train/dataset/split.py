import json
import random

# 读取原始 jsonl 文件
input_path = 'CADBench.jsonl'
with open(input_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

# 打乱数据顺序（保证随机性）
random.seed(42)  # 可复现
random.shuffle(data)

# 计算分割索引
split_index = int(len(data) * 0.7)
train_data = data[:split_index]
test_data = data[split_index:]

# 写入训练集
with open('CADBench_train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 写入测试集
with open('CADBench_test.jsonl', 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ 数据集拆分完成：训练集 {len(train_data)} 条，测试集 {len(test_data)} 条")
