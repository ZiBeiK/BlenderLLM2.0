# BlenderLLM2.0
## 模型训练
#### 利用BlendNet进行模型微调
```
cd train
python train.py
```
#### 利用CADBench进行增强训练
```
python retrain.py
```
---
## 示例运行
#### 调用API
```
cd BlenderLLM
python run_api.py "your instruction"
```
#### 使用trained model
```
python run_trained_model.py "your instruction"
```
#### 使用retrained model
```
python run_retrained_model.py "your instruction"
```
