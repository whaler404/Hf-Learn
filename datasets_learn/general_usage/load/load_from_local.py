from datasets import load_dataset

# csv
dataset = load_dataset("csv", data_files="my_file.csv")

# json
dataset = load_dataset("json", data_files="my_file.json")
# {"a": 1, "b": 2.0, "c": "foo", "d": false}
# {"a": 4, "b": -5.5, "c": null, "d": true}

# parquet
dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})

# arrow
dataset = load_dataset("arrow", data_files={'train': 'train.arrow', 'test': 'test.arrow'})

# Arrow is the file format used by 🤗 Datasets under the hood
from datasets import Dataset
dataset = Dataset.from_file("data.arrow")

# webdataset
path = "path/to/train/*.tar"
dataset = load_dataset("webdataset", data_files={"train": path}, split="train", streaming=True)

# multi processing
imagenet = load_dataset("timm/imagenet-1k-wds", num_proc=8)
ml_librispeech_spanish = load_dataset("facebook/multilingual_librispeech", "spanish", num_proc=8)