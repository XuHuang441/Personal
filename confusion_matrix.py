import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import confusion_matrix
import torch
import numpy as np


# 1. 加载你训练好的模型
model_path = "./esm2_phylum_finetuned_final"  # 你的模型保存路径
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=device, top_k=1)

# 2. 加载测试数据
test_file = "/hai/scratch/fangwu97/xu/Personal/data/esm2_phylum_dataset/test.tsv"
df_test = pd.read_csv(test_file, sep='\t')

# 为了展示快一点，我们只随机抽 1000 条测试数据（如果机器快可以用全部）
df_sample = df_test.sample(1000, random_state=42)

print("Running inference...")
# 获取预测结果
preds = classifier(df_sample['sequence'].tolist(), batch_size=32, truncation=True)
pred_labels = [p[0]['label'] for p in preds]
true_labels = df_sample['label'].tolist() # 注意：这里要是字符串标签(Proteobacteria等)

# 如果你的 test.tsv 里的 label 是数字，你需要用 id2label 映射回字符串，或者把 pred 也转数字
# 假设 label 列已经是字符串名称 (如 Proteobacteria)

# 3. 画混淆矩阵
labels = sorted(list(set(true_labels))) # 获取所有类别名
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

# 归一化 (看百分比而不是绝对数量)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('ESM2-150M Phylum Classification Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
print("Saved confusion_matrix.png")
plt.show()