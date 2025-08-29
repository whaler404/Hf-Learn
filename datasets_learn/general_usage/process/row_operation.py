from datasets import load_dataset
dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

# sort, numpy compatible
print(dataset["label"][:10])
sorted_dataset = dataset.sort("label")
print(sorted_dataset["label"][:10])

# shuffle
shuffled_dataset = sorted_dataset.shuffle(seed=42)
print(shuffled_dataset["label"][:10])
# fast approximate shuffling  
iterable_dataset = dataset.to_iterable_dataset(num_shards=128)
shuffled_iterable_dataset = iterable_dataset.shuffle(seed=42, buffer_size=1000)

# select rows
small_dataset = dataset.select([0, 10, 20, 30, 40, 50])
print(len(small_dataset))

# filter rows
start_with_ar = dataset.filter(lambda example: example["sentence1"].startswith("Ar"))
print(len(start_with_ar))
print(start_with_ar["sentence1"])
# filter by indices
even_dataset = dataset.filter(lambda example, idx: idx % 2 == 0, with_indices=True)
print(len(even_dataset))
print(len(dataset) / 2)

# shard
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/imdb", split="train")
print(dataset)
# Dataset({
#     features: ['text', 'label'],
#     num_rows: 25000
# })
chunk = dataset.shard(num_shards=4, index=0)
print(chunk)
# Dataset({
#     features: ['text', 'label'],
#     num_rows: 6250
# })