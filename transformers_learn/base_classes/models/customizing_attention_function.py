from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import torch

model_id = "openai-community/gpt2"

# # 1. customize the attention function
# def my_new_sdpa(*args, **kwargs):
#     print("I just entered the attention computation")
#     return sdpa_attention_forward(*args, **kwargs)

# AttentionInterface.register("my_new_sdpa", my_new_sdpa)

# model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="my_new_sdpa")
# # Try running the forward with the new attention function
# model(torch.ones(1, 5, dtype=int))

# # 2. cover the original attention implementation
# # Back to use original sdpa implementation
# model = AutoModelForCausalLM.from_pretrained(model_id)
# model.config._attn_implementation = "sdpa"

# model(torch.ones(1, 5, dtype=int))

# 3. Customize the attention function with new kwargs
from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import torch
from typing import Optional, Tuple

def custom_attention(
    module: torch.nn.Module,  # required arg
    query: torch.Tensor,  # required arg
    key: torch.Tensor,  # required arg
    value: torch.Tensor,  # required arg
    attention_mask: Optional[torch.Tensor],  # required arg
    a_new_kwargs = None,  # You can now add as many kwargs as you need
    another_new_kwargs = None,  # You can now add as many kwargs as you need
    **kwargs,  # You need to accept **kwargs as models will pass other args
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    attn_output, attn_weights = sdpa_attention_forward(
        module, query, key, value, attention_mask, **kwargs
    )
    # add linear projection to attn_output with the same shape
    linear_proj = torch.nn.Linear(attn_output.shape[-1], attn_output.shape[-1])
    attn_output = linear_proj(attn_output)
    return attn_output, attn_weights  # attn_weights are optional here

AttentionInterface.register("custom", custom_attention)

model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="custom")
# Forward pass with the new kwargs
result = model(torch.ones(1, 5, dtype=int), a_new_kwargs=..., another_new_kwargs=...)
print(result)