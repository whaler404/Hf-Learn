from datasets import load_dataset

# faster
dataset = load_dataset("ethz/food101")
iterable_dataset = dataset.to_iterable_dataset()

# slower
iterable_dataset = load_dataset("ethz/food101", streaming=True)

# enable fast parallel loading with a PyTorch DataLoader.
import torch
from datasets import load_dataset

dataset = load_dataset("ethz/food101")
iterable_dataset = dataset.to_iterable_dataset(num_shards=64) # shard the dataset
iterable_dataset = iterable_dataset.shuffle(buffer_size=10_000)  # shuffles the shards order and use a shuffle buffer when you start iterating
dataloader = torch.utils.data.DataLoader(iterable_dataset, num_workers=4)  # assigns 64 / 4 = 16 shards from the shuffled list of shards to each worker when you start iterating
