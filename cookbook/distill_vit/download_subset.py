from datasets import load_dataset, Dataset

# 加载单个分片
dataset:Dataset = load_dataset(
    "parquet",
    data_files=[
        "datasets/datasets--ILSVRC--imagenet-1k/data/train-00000-of-00294.parquet",
        "datasets/datasets--ILSVRC--imagenet-1k/data/train-00001-of-00294.parquet",
    ],
)["train"]

dataset = dataset.select([i for i in range(200)])

print(dataset)
# 输出：Dataset({
#   features: ['image', 'label'],
#   num_rows: ~5000  # 每片约几千到上万条样本
# })

# Step 1: 先拆出测试集（20%）
split = dataset.train_test_split(test_size=0.2, seed=42)
train_valid = split["train"]
test = split["test"]

# Step 2: 再从训练部分拆出验证集（10% of train）
split2 = train_valid.train_test_split(test_size=0.1, seed=42)
train = split2["train"]
valid = split2["test"]

print(train.num_rows, valid.num_rows, test.num_rows)

from datasets import DatasetDict
dataset_splits = DatasetDict({
    "train": train,
    "validation": valid,
    "test": test
})

import os

save_dir = "datasets/imagenet-1k-subset-tiny/data"
os.makedirs(save_dir, exist_ok=True)

dataset_splits["train"].to_parquet(os.path.join(save_dir, "imagenet_train.parquet"))
dataset_splits["validation"].to_parquet(os.path.join(save_dir, "imagenet_val.parquet"))
dataset_splits["test"].to_parquet(os.path.join(save_dir, "imagenet_test.parquet"))
