# shuffle
from datasets import load_dataset
dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)

# reshuffle the dataset after each epoch
epochs = 5
shuffled_dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
for epoch in range(epochs):
    shuffled_dataset.set_epoch(epoch)
    for example in shuffled_dataset:
        ...

# returns the first n examples in a dataset
dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
dataset_head = dataset.take(2)
print(list(dataset_head))
# [{'text': "How AP reported in all formats from tor...},
#  {'text': 'Did you know you have two little yellow...}]

# omits the first n examples in a dataset and returns the remaining examples
train_dataset = shuffled_dataset.skip(1000)

# shard
from datasets import load_dataset
dataset = load_dataset("amazon_polarity", split="train", streaming=True)
print(dataset)
# IterableDataset({
#     features: ['label', 'title', 'content'],
#     num_shards: 4
# })
chunk = dataset.shard(num_shards=2, index=0)
print(chunk)
# IterableDataset({
#     features: ['label', 'title', 'content'],
#     num_shards: 2
# })

# interleave
from datasets import interleave_datasets
es_dataset = load_dataset('allenai/c4', 'es', split='train', streaming=True)
fr_dataset = load_dataset('allenai/c4', 'fr', split='train', streaming=True)
# combine an IterableDataset with other datasets
multilingual_dataset = interleave_datasets([es_dataset, fr_dataset])
print(list(multilingual_dataset.take(2)))
# [{'text': 'Comprar Zapatillas para niña en chancla con goma por...'},
#  {'text': 'Le sacre de philippe ier, 23 mai 1059 - Compte Rendu...'}]

# Define sampling probabilities
multilingual_dataset_with_oversampling = interleave_datasets([es_dataset, fr_dataset], probabilities=[0.8, 0.2], seed=42)
print(list(multilingual_dataset_with_oversampling.take(2)))
# [{'text': 'Comprar Zapatillas para niña en chancla con goma por...'},
#  {'text': 'Chevrolet Cavalier Usados en Bogota - Carros en Vent...'}]

