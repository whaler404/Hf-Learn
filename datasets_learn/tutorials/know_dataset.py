from datasets import load_dataset

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")

print(type(dataset))
# <class 'datasets.arrow_dataset.Dataset'>
print(dataset[-1])
# {'text': 'things really get weird , though not particularly scary : the movie is all portent and no content .', 'label': 0}

print(type(dataset["text"]))
# list
print(dataset["text"][0])

# slicing
print(dataset[:5]['label'])  # first 5 examples
# [1, 1, 1, 1, 1]

# from datasets import load_dataset

# # streaming from the Hub
# iterable_dataset = load_dataset("ethz/food101", split="train", streaming=True)

from datasets import load_dataset

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
iterable_dataset = dataset.to_iterable_dataset()
# next(iter(iterable_dataset["label"]))

sub_dataset = iterable_dataset.take(3)
print(sub_dataset)
# IterableDataset({
#     features: ['text', 'label'],
#     num_shards: 1
# })

print(type(sub_dataset))
# <class 'datasets.iterable_dataset.IterableDataset'>

