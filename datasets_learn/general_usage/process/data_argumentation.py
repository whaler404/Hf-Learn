from random import randint
from transformers import pipeline

from datasets import load_dataset
dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

fillmask = pipeline("fill-mask", model="google-bert/bert-base-cased")
mask_token = fillmask.tokenizer.mask_token
smaller_dataset = dataset.filter(lambda e, i: i<10, with_indices=True)

def augment_data(examples):
    outputs = []
    for sentence in examples["sentence1"]:
        words = sentence.split(' ')
        K = randint(1, len(words)-1)
        masked_sentence = " ".join(words[:K]  + [mask_token] + words[K+1:])
        predictions = fillmask(masked_sentence)
        augmented_sequences = [predictions[i]["sequence"] for i in range(3)]
        outputs += [sentence] + augmented_sequences
    return {"data": outputs}

augmented_dataset = smaller_dataset.map(augment_data, batched=True, remove_columns=dataset.column_names, batch_size=8)
print(augmented_dataset[:9]["data"])
# ['Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
#  'Amrozi accused his brother, whom he called " the witness ", of deliberately distorting his evidence.',
#  'Amrozi accused his brother, whom he called " the witness ", for deliberately distorting his evidence.',
#  'Amrozi accused his brother, whom he called " the witness ", with deliberately distorting his evidence.',
#  "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .",
#  "Yucaipa owned Dominick ' s before selling the chain to Safeway in 1998 for $ 2. 5 billion.",
#  "Yucaipa owned Dominick ' s before selling the chain to Safeway in 1998 for $ 2. 5 billion ;",
#  "Yucaipa owned Dominick ' s before selling the chain to Safeway in 1998 for $ 2. 5 billion!",
#  'They had published an advertisement on the Internet on June 10 , offering the cargo for sale , he added .']