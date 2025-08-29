import datasets

# Concatenate a train and test
train_test_ds = datasets.load_dataset("ajibawa-2023/General-Stories-Collection", split="train+test")

# select specific rows of train split
train_10_20_ds = datasets.load_dataset("ajibawa-2023/General-Stories-Collection", split="train[10:20]")

# select a percentage of a split 
train_10pct_ds = datasets.load_dataset("ajibawa-2023/General-Stories-Collection", split="train[:10%]")

# Select a combination of percentages from each split
train_10_80pct_ds = datasets.load_dataset("ajibawa-2023/General-Stories-Collection", split="train[:10%]+train[-80%:]")

# create cross-validated splits
val_ds = datasets.load_dataset("ajibawa-2023/General-Stories-Collection", split=[f"train[{k}%:{k+10}%]" for k in range(0, 100, 10)])
train_ds = datasets.load_dataset("ajibawa-2023/General-Stories-Collection", split=[f"train[:{k}%]+train[{k+10}%:]" for k in range(0, 100, 10)])