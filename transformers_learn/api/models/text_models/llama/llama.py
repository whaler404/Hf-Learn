from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = LlamaForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

print(model)

from torchinfo import summary
summary(model)


import torch
result = model(input_ids=torch.arange(1, 10).view(1, -1), labels=torch.arange(1, 10).view(1, -1))
print(result.loss)
print(result.logits.shape)
# print(result.hidden_states)