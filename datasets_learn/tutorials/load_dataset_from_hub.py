# load the dataset builder and inspect its metadata
from datasets import load_dataset_builder
ds_builder = load_dataset_builder("cornell-movie-review-data/rotten_tomatoes")

print(ds_builder.info.description)

print(ds_builder.info.features)
# {'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['neg', 'pos'], id=None)}

# load the dataset
from datasets import load_dataset

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 8530
#     })
#     validation: Dataset({
#         features: ['text', 'label'],
#         num_rows: 1066
#     })
#     test: Dataset({
#         features: ['text', 'label'],
#         num_rows: 1066
#     })
# })

# a split is a specific subset of the dataset
from datasets import get_dataset_split_names

splits = get_dataset_split_names("cornell-movie-review-data/rotten_tomatoes")

print(splits)
# ['train', 'validation', 'test']

# datasets contains several subsets, called configurations
from datasets import get_dataset_config_names

configs = get_dataset_config_names("nyu-mll/glue")

print(configs)
# ['ax', 'cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

# load a specific configuration of a dataset
from datasets import load_dataset

mindsFR = load_dataset("nyu-mll/glue", "ax")

print(mindsFR)
# DatasetDict({
#     test: Dataset({
#         features: ['premise', 'hypothesis', 'label', 'idx'],
#         num_rows: 1104
#     })
# })