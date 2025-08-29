from datasets import load_dataset

# specify the dataset version
dataset = load_dataset(
  "lhoestq/custom_squad",
  revision="main"  # tag name, or branch name, or commit hash
)

# map data files to splits like train, validation and test
data_files = {"train": "train.csv", "test": "test.csv"}
dataset = load_dataset("namespace/your_dataset_name", data_files=data_files)

# load a specific subset of the files 
c4_subset = load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz")

c4_subset = load_dataset("allenai/c4", data_dir="en")

# map a data file to a specific split
data_files = {"validation": "en/c4-validation.*.json.gz"}
c4_validation = load_dataset("allenai/c4", data_files=data_files, split="validation")