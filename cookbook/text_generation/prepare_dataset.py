from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        # 不需要固定长度，动态 pad 更高效
        padding=False,  
        return_attention_mask=True,
    )


def prepare_dataset(dataset_path="datasets/datasets--roneneldan--TinyStories--tiny", batch_size=8, num_workers=4):
    # 1️⃣ 加载 parquet 数据
    train_dataset = load_dataset("parquet", data_files=f"{dataset_path}/train*.parquet", split="train")
    val_dataset = load_dataset("parquet", data_files=f"{dataset_path}/validation*.parquet", split="train")

    # 2️⃣ 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3️⃣ Tokenize
    train_dataset = train_dataset.map(lambda ex: tokenize_function(ex, tokenizer), batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(lambda ex: tokenize_function(ex, tokenizer), batched=True, remove_columns=val_dataset.column_names)

    # 4️⃣ 设置 PyTorch 格式
    train_dataset.set_format(type="torch")
    val_dataset.set_format(type="torch")

    # 5️⃣ 使用 Hugging Face 自带的 DataCollator
    # 它会在 batch 时自动 pad + 复制 labels
    # DataCollatorForLanguageModeling 是 Hugging Face Transformers 的数据整理器，用于在组装训练批次时为语言模型任务（尤其是掩码语言建模，MLM）执行动态掩码和对齐填充，并生成训练所需的 labels
    # mlm (bool): 是否进行 MLM 动态掩码，默认 True
    # return_tensors (str): 返回张量类型，默认 pt（PyTorch）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 对于自回归模型，如 SmolLM, GPT 类，应设为 False
    )

    # 6️⃣ 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator, num_workers=num_workers)

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    train_loader, val_loader, tokenizer = prepare_dataset()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 👀 测试一个 batch
    batch = next(iter(train_loader))
    print({k: v.shape for k, v in batch.items()})
