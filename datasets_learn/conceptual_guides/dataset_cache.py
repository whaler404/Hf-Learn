from datasets import Dataset

dataset1 = Dataset.from_dict({"a": [0, 1, 2]})
dataset2 = dataset1.map(lambda x: {"a": x["a"] + 1})
print(dataset1._fingerprint, dataset2._fingerprint)
# 889f7244aaee43a3 108a96996544930b

from datasets.fingerprint import Hasher
my_func = lambda example: {"length": len(example["text"])}
print(Hasher.hash(my_func))