from datasets import load_dataset
dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

# rename
print(dataset)
# Dataset({
#     features: ['sentence1', 'sentence2', 'label', 'idx'],
#     num_rows: 3668
# })
dataset = dataset.rename_column("sentence1", "sentenceA")
dataset = dataset.rename_column("sentence2", "sentenceB")
print(dataset)
# Dataset({
#     features: ['sentenceA', 'sentenceB', 'label', 'idx'],
#     num_rows: 3668
# })

# remove
dataset = dataset.remove_columns("label")
print(dataset)
# Dataset({
#     features: ['sentence1', 'sentence2', 'idx'],
#     num_rows: 3668
# })
dataset = dataset.remove_columns(["sentence1", "sentence2"])
print(dataset)
# Dataset({
#     features: ['idx'],
#     num_rows: 3668
# })

# select columns
dataset = dataset.select_columns(['sentence1', 'sentence2', 'idx'])
print(dataset)
# Dataset({
#     features: ['sentence1', 'sentence2', 'idx'],
#     num_rows: 3668
# })
dataset = dataset.select_columns('idx')
print(dataset)
# Dataset({
#     features: ['idx'],
#     num_rows: 3668
# })

# cast
print(dataset.features)
# {'sentence1': Value('string'),
# 'sentence2': Value('string'),
# 'label': ClassLabel(names=['not_equivalent', 'equivalent']),
# 'idx': Value('int32')}

from datasets import ClassLabel, Value
new_features = dataset.features.copy()
new_features["label"] = ClassLabel(names=["negative", "positive"])
new_features["idx"] = Value("int64")
dataset = dataset.cast(new_features)
print(dataset.features)
# {'sentence1': Value('string'),
# 'sentence2': Value('string'),
# 'label': ClassLabel(names=['negative', 'positive']),
# 'idx': Value('int64')}

dataset = dataset.cast_column("label", ClassLabel(names=['0', '1']))
print(dataset.features)
# {'sentence1': Value('string'),
# 'sentence2': Value('string'),
# 'label': ClassLabel(names=['0', '1']),
# 'idx': Value('int64')}

# flatten
from datasets import load_dataset
dataset = load_dataset("rajpurkar/squad", split="train")
print(dataset.features)
# {'id': Value('string'),
# 'title': Value('string'),
# 'context': Value('string'),
# 'question': Value('string'),
# 'answers': Sequence({
#     'text': Value('string'),
#     'answer_start': Value('int32')
# })}
flat_dataset = dataset.flatten()
print(flat_dataset.features)
# {'id': Value('string'),
# 'title': Value('string'),
# 'context': Value('string'),
# 'question': Value('string'),
# 'answers.text': Value('string'),
# 'answers.answer_start': Value('int32')}