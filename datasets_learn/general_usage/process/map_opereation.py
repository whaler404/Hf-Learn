from datasets import load_dataset
dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

def add_prefix(example):
    example["sentence1"] = 'My sentence: ' + example["sentence1"]
    return example

small_dataset = dataset.select([0, 10, 20, 30, 40, 50])

updated_dataset = small_dataset.map(add_prefix)
print(updated_dataset["sentence1"][:5])
# ['My sentence: Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
# "My sentence: Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .",
# 'My sentence: They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .',
# 'My sentence: Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 .',
# ]

# Specify the column to remove 
updated_dataset = dataset.map(lambda example: {"new_sentence": example["sentence1"]}, remove_columns=["sentence1"])
print(updated_dataset.column_names)
# ['sentence2', 'label', 'idx', 'new_sentence']

# with indices
updated_dataset = dataset.map(lambda example, idx: {"sentence2": f"{idx}: " + example["sentence2"]}, with_indices=True)
print(updated_dataset["sentence2"][:5])
# ['0: Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
#  "1: Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 .",
#  "2: On June 10 , the ship 's owners had published an advertisement on the Internet , offering the explosives for sale .",
#  '3: Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 .',
#  '4: PG & E Corp. shares jumped $ 1.63 or 8 percent to $ 21.03 on the New York Stock Exchange on Friday .'
# ]

# Multiprocessing significantly speeds up processing by parallelizing processes on the CPU
updated_dataset = dataset.map(lambda example, idx: {"sentence2": f"{idx}: " + example["sentence2"]}, with_indices=True, num_proc=4)

# batch processing

# split long examples

# data augmentation

# async processing

# process multiple splits

from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("wheeler404/qwen2-tiny")
dataset = load_dataset('nyu-mll/glue', 'mrpc')
encoded_dataset = dataset.map(lambda examples: tokenizer(examples["sentence1"]), batched=True)
print(encoded_dataset["train"][0])
# {'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
# 'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
# 'label': 1,
# 'idx': 0,
# 'input_ids': [  101,  7277,  2180,  5303,  4806,  1117,  1711,   117,  2292, 1119,  1270,   107,  1103,  7737,   107,   117,  1104,  9938, 4267, 12223, 21811,  1117,  2554,   119,   102],
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# }