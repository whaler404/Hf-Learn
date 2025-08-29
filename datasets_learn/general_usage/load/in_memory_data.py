from datasets import Dataset

# python dictionary to Dataset
my_dict = {"a": [1, 2, 3]}
dataset = Dataset.from_dict(my_dict)

# python list of dictionaries to Dataset
my_list = [{"a": 1}, {"a": 2}, {"a": 3}]
dataset = Dataset.from_list(my_list)

# python generator to Dataset
def my_gen():
    for i in range(1, 4):
        yield {"a": i}
dataset = Dataset.from_generator(my_gen)

# loading data larger than memory
from datasets import IterableDataset
def gen(shards):
    for shard in shards:
        with open(shard) as f:
            for line in f:
                yield {"line": line}
shards = [f"data{i}.txt" for i in range(32)]
ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
ds = ds.shuffle(seed=42, buffer_size=10_000)  # shuffles the shards order + uses a shuffle buffer
from torch.utils.data import DataLoader
dataloader = DataLoader(ds.with_format("torch"), num_workers=4)  # give each worker a subset of 32/4=8 shards

# pandas DataFrame to Dataset
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3]})
dataset = Dataset.from_pandas(df)