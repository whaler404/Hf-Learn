# shuffle
from datasets import load_dataset
dataset = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)

seed, buffer_size = 42, 10_000
dataset = dataset.shuffle(seed, buffer_size=buffer_size)

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling, AutoTokenizer
from tqdm import tqdm

dataset = dataset.with_format("torch")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataloader = DataLoader(dataset, collate_fn=DataCollatorForLanguageModeling(tokenizer))

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
model.train().to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
for epoch in range(3):
    dataset.set_epoch(epoch)
    for i, batch in enumerate(tqdm(dataloader, total=5)):
        if i == 5:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"loss: {loss}")