import aiohttp
import asyncio
from huggingface_hub import get_token
sem = asyncio.Semaphore(20)  # max number of simultaneous queries

async def query_model(model, prompt):
    api_url = f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {get_token()}", "Content-Type": "application/json"}
    json = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 20, "seed": 42}
    async with sem, aiohttp.ClientSession() as session, session.post(api_url, headers=headers, json=json) as response:
        output = await response.json()
        return {"Output": output["choices"][0]["message"]["content"]}
    
from datasets import load_dataset
ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
model = "microsoft/Phi-3-mini-4k-instruct"
prompt = 'What is this text mainly about ? Here is the text:\n\n```\n{Problem}\n```\n\nReply using one or two words max, e.g. "The main topic is Linear Algebra".'
async def get_topic(example):
    return await query_model(model, prompt.format(Problem=example['Problem']))
ds = ds.map(get_topic)
print(ds[0])
# {'ID': '2024-II-4',
#  'Problem': 'Let $x,y$ and $z$ be positive real numbers that...',
#  'Solution': 'Denote $\\log_2(x) = a$, $\\log_2(y) = b$, and...,
#  'Answer': 33,
#  'Output': 'The main topic is Logarithms.'}