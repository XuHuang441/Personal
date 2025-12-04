import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb

# ================= 配置路径 =================
TRAIN_FILE = "/Users/huangxu/Library/CloudStorage/OneDrive-个人/phd/PKU/Yang Bai/crbc/esm2_phylum_dataset/train.tsv"
TEST_FILE  = "/Users/huangxu/Library/CloudStorage/OneDrive-个人/phd/PKU/Yang Bai/crbc/esm2_phylum_dataset/test.tsv"
MODEL_CHECKPOINT = "facebook/esm2_t30_150M_UR50D"

# WandB 项目名称
WANDB_PROJECT = "esm2-phylum-finetune"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT

def main():
    # 1. 读取数据并处理标签
    print(f"Loading data from {TRAIN_FILE} and {TEST_FILE}...")
    
    # 假设 tsv 是用 tab 分隔的
    df_train = pd.read_csv(TRAIN_FILE, sep='\t')
    df_test = pd.read_csv(TEST_FILE, sep='\t')

    # 检查列名，如果不叫 'label' 和 'sequence'，请在这里修改
    # 假设你的列名是 'Phylum' 和 'sequence'，这里做一个重命名以防万一
    # 如果你的列名已经是 'label'，这行代码不会报错
    if 'Phylum' in df_train.columns:
        df_train = df_train.rename(columns={'Phylum': 'label'})
        df_test = df_test.rename(columns={'Phylum': 'label'})
    
    # === 核心步骤：将文本标签转换为数字 ID ===
    # 获取所有唯一的标签类别并排序，保证每次运行顺序一致
    unique_labels = sorted(df_train['label'].unique().tolist())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    
    print(f"Detected {len(unique_labels)} classes: {label2id}")
    
    # 映射标签
    df_train['label'] = df_train['label'].map(label2id)
    df_test['label'] = df_test['label'].map(label2id)

    # 转换为 HuggingFace Dataset
    train_ds = Dataset.from_pandas(df_train)
    test_ds = Dataset.from_pandas(df_test)

    # 2. Tokenizer 处理
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize_function(examples):
        # 截断长度设为 1024 (ESM2 的上限)，H100 显存足够吃下
        return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=1024)

    print("Tokenizing data...")
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)

    # 3. 加载模型
    # num_labels 自动根据刚才检测到的类别数设定
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )

    # 4. 定义评估指标
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # 5. 训练参数 (针对 H100 优化)
    training_args = TrainingArguments(
        output_dir="./esm2_results",
        learning_rate=2e-5,              # 经典的微调学习率
        per_device_train_batch_size=32,  # H100 80G 可以尝试 32 甚至 64 (如果 OOM 就降到 16)
        per_device_eval_batch_size=32,
        num_train_epochs=3,              # 3个 Epoch 足够收敛
        weight_decay=0.01,
        evaluation_strategy="epoch",     # 每个 Epoch 结束后评估一次
        save_strategy="epoch",           # 每个 Epoch 保存一次模型
        save_total_limit=2,              # 只保留最近的2个模型，节省硬盘
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,                # WandB 每 50 步记录一次 Loss
        report_to="wandb",               # 开启 WandB 报告
        
        # === H100 专属加速设置 ===
        bf16=True,                       # H100 支持 BFloat16，比 fp16 更稳更好
        tf32=True,                       # 开启 TensorFloat-32 加速矩阵乘法
        dataloader_num_workers=8,        # 加快数据加载
        run_name="esm2-150m-h100-run"    # WandB 里的 Run 名字
    )

    # 6. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    # 7. 开始训练
    print("Starting training...")
    trainer.train()

    # 8. 训练结束，保存最终模型
    trainer.save_model("./esm2_phylum_finetuned_final")
    tokenizer.save_pretrained("./esm2_phylum_finetuned_final")
    print("Training finished. Model saved to ./esm2_phylum_finetuned_final")

if __name__ == "__main__":
    main()
