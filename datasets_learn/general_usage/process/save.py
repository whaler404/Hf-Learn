from datasets import Dataset, interleave_datasets

d1 = Dataset.from_dict({"a": [0, 1, 2]})
d2 = Dataset.from_dict({"a": [10, 11, 12, 13]})
d3 = Dataset.from_dict({"a": [20, 21, 22]})
dataset = interleave_datasets([d1, d2, d3], stopping_strategy="all_exhausted")
print(dataset["a"])
# [0, 10, 20, 1, 11, 21, 2, 12, 22, 0, 13, 20]

dataset = dataset.with_format(type="torch")
print(dataset.format)
# {'type': 'torch', 'format_kwargs': {}, 'columns': [...], 'output_all_columns': False}
ds = Dataset.from_dict({"text": ["foo", "bar"], "tokens": [[0, 1, 2], [3, 4, 5]]})
ds = ds.with_format("torch")
print(ds[0])
# {'text': 'foo', 'tokens': tensor([0, 1, 2])}
print(ds[:2])
# {'text': ['foo', 'bar'],
#  'tokens': tensor([[0, 1, 2],
#          [3, 4, 5]])}

# use multiple processes to upload it in parallel
dataset.push_to_hub("username/my_dataset", num_proc=8)

# save it locally in Arrow format on disk.
dataset.save_to_disk("path/of/my/dataset/directory")
# later
from datasets import load_from_disk
reloaded_dataset = load_from_disk("path/of/my/dataset/directory")

# export to CSV
dataset.to_csv("path/of/my/dataset.csv")