import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, AutoConfig
from datasets import Dataset
from sklearn.metrics import accuracy_score

# ================= 配置 =================
TEST_FILE = "/hai/scratch/fangwu97/xu/Personal/data/esm2_phylum_dataset/test.tsv"
FINETUNED_MODEL_PATH = "./esm2_phylum_finetuned_final"
BASE_MODEL_NAME = "/hai/scratch/fangwu97/xu/cache/esm2_t30_150M_UR50D"


def main():
    # 1. 准备数据
    print("Loading Test Data...")
    df_test = pd.read_csv(TEST_FILE, sep='\t')

    # 从微调后的模型配置里读取 label map，保证一一对应
    config = AutoConfig.from_pretrained(FINETUNED_MODEL_PATH)
    label2id = config.label2id
    id2label = config.id2label

    # --- 修正点开始：处理列名 ---
    # 1. 确保原始 label 列名正确
    if 'Phylum' in df_test.columns:
        df_test = df_test.rename(columns={'Phylum': 'label_name'})
    elif 'label' in df_test.columns:
        df_test = df_test.rename(columns={'label': 'label_name'})

    # 2. 生成数字 ID，并强制命名为 'label' (Trainer 只认这个名字)
    # 注意：df_test['label_name'] 必须是字符串类型
    df_test['label'] = df_test['label_name'].map(label2id)

    # 3. 检查是否有 NaN (以防 map 失败)
    if df_test['label'].isnull().any():
        print("Warning: Some labels could not be mapped! Dropping them.")
        df_test = df_test.dropna(subset=['label'])

    # 强制转换为整数类型
    df_test['label'] = df_test['label'].astype(int)
    # --- 修正点结束 ---

    # 为了演示速度，只取 500 条测试
    df_sample = df_test.sample(500, random_state=42)

    # 创建 Dataset，只保留 sequence 和 label (数字)
    test_ds = Dataset.from_pandas(df_sample[['sequence', 'label']])

    # 2. 准备 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def tokenize(examples):
        return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=1024)

    print("Tokenizing...")
    encoded_test = test_ds.map(tokenize, batched=True)

    # ================= 模型 1: 未微调的原始模型 (Base) =================
    print("\n--- Evaluating Base Model (Pre-trained Only) ---")
    # 加载 Base 模型，分类头随机初始化
    model_base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    trainer_base = Trainer(model=model_base)
    preds_base = trainer_base.predict(encoded_test)
    acc_base = accuracy_score(df_sample['label'], np.argmax(preds_base.predictions, axis=1))
    print(f"Base Model Accuracy: {acc_base:.4f}")

    # ================= 模型 2: 你的微调模型 (Fine-tuned) =================
    print("\n--- Evaluating Fine-tuned Model (Yours) ---")
    model_finetuned = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_PATH)
    trainer_finetuned = Trainer(model=model_finetuned)
    preds_finetuned = trainer_finetuned.predict(encoded_test)
    acc_finetuned = accuracy_score(df_sample['label'], np.argmax(preds_finetuned.predictions, axis=1))
    print(f"Fine-tuned Model Accuracy: {acc_finetuned:.4f}")

    # ================= 3. 画出震撼的对比图 =================
    plt.figure(figsize=(8, 6))
    models = ['Base Model (Random Head)', 'Your Fine-tuned Model']
    accuracies = [acc_base, acc_finetuned]
    colors = ['grey', '#1f77b4']

    bar_plot = sns.barplot(x=models, y=accuracies, palette=colors)
    plt.title('Effect of Fine-tuning on Phylum Classification', fontsize=15)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)  # 留点空间给数字

    # 在柱子上标数值
    for i, v in enumerate(accuracies):
        bar_plot.text(i, v + 0.02, f"{v:.2%}", ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig("finetune_comparison.png", dpi=300)
    print("\nComparison plot saved to finetune_comparison.png")
    # plt.show() # 如果在服务器上，这行通常注释掉


if __name__ == "__main__":
    main()