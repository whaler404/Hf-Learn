from datasets import Dataset

iterable_dataset = Dataset.from_dict({"a": range(6)}).to_iterable_dataset(num_shards=3)
for idx, example in enumerate(iterable_dataset):
    print(example)
    if idx == 2:
        state_dict = iterable_dataset.state_dict()
        print(state_dict)
        print("checkpoint")
        break
# {'a': 0}
# {'a': 1}
# {'a': 2}
# {
#   "examples_iterable": {
#     "examples_iterable": {
#       "shard_idx": 1,
#       "shard_example_idx": 2,
#       "type": "ArrowExamplesIterable"
#     },
#     "previous_state": {
#       "shard_idx": 0,
#       "shard_example_idx": 0,
#       "type": "ArrowExamplesIterable"
#     },
#     "batch_idx": 3,
#     "num_chunks_since_previous_state": 3,
#     "cropped_chunk_length": 0,
#     "type": "RebatchedArrowExamplesIterable"
#   },
#   "epoch": 0
# }
# checkpoint
# restart from checkpoint
iterable_dataset.load_state_dict(state_dict)
print(f"restart from checkpoint")
for example in iterable_dataset:
    print(example)
# {'a': 3}
# {'a': 4}
# {'a': 5}