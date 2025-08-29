from datasets import load_dataset
dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

# batch processing

# Splits the sentence1 field into chunks of 50 characters.
def chunk_examples(examples):
    chunks = []
    for sentence in examples["sentence1"]:
        chunks += [sentence[i:i + 50] for i in range(0, len(sentence), 50)]
    return {"chunks": chunks}

print(dataset)
# Dataset({
#  features: ['sentence1', 'sentence2', 'label', 'idx'],
#  num_rows: 3668
# })

# Stacks all the chunks together to create the new dataset.
chunked_dataset = dataset.map(chunk_examples, batched=True, remove_columns=dataset.column_names)
print(chunked_dataset)
# Dataset({
#     features: ['chunks'],
#     num_rows: 10470
# })
