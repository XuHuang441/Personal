import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 1. 配置
model_path = "./esm2_phylum_finetuned_final"
test_file = "/hai/scratch/fangwu97/xu/Personal/data/esm2_phylum_dataset/test.tsv"

# 2. 加载模型
print("Loading model...")
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=device)

# 3. 读取数据
df = pd.read_csv(test_file, sep='\t')
# 确保包含 True Label
# 假设你的 label 列已经是字符串 (Proteobacteria 等)，如果是数字需转换

# 4. 每个类别抽 1 个样本
unique_labels = df['label'].unique()
samples = []

for label in unique_labels:
    # 随机抽一条该类别的序列
    row = df[df['label'] == label].sample(1).iloc[0]
    samples.append(row)

# 5. 生成对比测试用的 Prompt
print("\n" + "=" * 20 + " 对比实验素材 " + "=" * 20)

for i, row in enumerate(samples):
    seq = row['sequence']
    true_label = row['label']

    # 使用你的模型进行预测
    pred = classifier(seq)[0]
    my_model_pred = pred['label']
    my_model_conf = pred['score']

    print(f"\n--- [Case {i + 1}: True Label = {true_label}] ---")
    print(f"【你的 ESM2 模型】预测: {my_model_pred} (置信度: {my_model_conf:.4f})")

    print(f"\n【复制给 GPT 的 Prompt】:")
    print(f"""
I have a protein sequence from a bacteria. Please classify it into one of these phylums: 
[Proteobacteria, Actinobacteriota, Firmicutes, Bacteroidota].
Only output the phylum name.

Sequence:
{seq}
    """)
    print("-" * 60)