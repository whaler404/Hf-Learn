from typing import Callable, Optional, Tuple
import torch
from torch import nn

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, LlamaAttention
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    print(f"[eager_attention_forward] key_states shape: {key_states.shape}")
    value_states = repeat_kv(value, module.num_key_value_groups)
    print(f"[eager_attention_forward] value_states shape: {value_states.shape}")

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    print(f"[eager_attention_forward] attn_weights shape before mask: {attn_weights.shape}")
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    print(f"[eager_attention_forward] attn_output shape before transpose: {attn_output.shape}")
    attn_output = attn_output.transpose(1, 2).contiguous()
    print(f"[eager_attention_forward] attn_output shape after transpose: {attn_output.shape}")

    return attn_output, attn_weights

class MyLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        print(f"[LlamaAttention] input hidden_states shape: {hidden_states.shape}")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        print(f"[LlamaAttention] query_states shape: {query_states.shape}")
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        print(f"[LlamaAttention] key_states shape: {key_states.shape}")
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        print(f"[LlamaAttention] value_states shape: {value_states.shape}")

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        print(f"[LlamaAttention] query_states after RoPE shape: {query_states.shape}")
        print(f"[LlamaAttention] key_states after RoPE shape: {key_states.shape}")

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            print(f"[LlamaAttention] key_states after cache shape: {key_states.shape}")
            print(f"[LlamaAttention] value_states after cache shape: {value_states.shape}")

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        print(f"[LlamaAttention] attn_output shape: {attn_output.shape}")
        print(f"[LlamaAttention] attn_weights shape: {attn_weights.shape if attn_weights is not None else None}")

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        print(f"[LlamaAttention] attn_output after reshape shape: {attn_output.shape}")
        attn_output = self.o_proj(attn_output)
        print(f"[LlamaAttention] attn_output after o_proj shape: {attn_output.shape}")
        return attn_output, attn_weights

import torch
from transformers import LlamaConfig

# 构造 Llama 配置
config = LlamaConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# 构造输入
batch_size = 2
seq_len = 16
hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
cos = torch.randn(batch_size, seq_len, config.hidden_size // config.num_attention_heads)
sin = torch.randn(batch_size, seq_len, config.hidden_size // config.num_attention_heads)
position_embeddings = (cos, sin)
attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len)

# 实例化自定义 LlamaAttention
llama_attention = MyLlamaAttention(config, layer_idx=0)
# 前向测试
output, attn_weights = llama_attention(
    hidden_states=hidden_states,
    position_embeddings=position_embeddings,
    attention_mask=attention_mask,
)
print("output shape:", output.shape)
if attn_weights is not None:
    print("attn_weights shape:", attn_weights.shape)

# [LlamaAttention] input hidden_states shape: torch.Size([2, 16, 576])
# [LlamaAttention] query_states shape: torch.Size([2, 9, 16, 64])
# [LlamaAttention] key_states shape: torch.Size([2, 3, 16, 64])
# [LlamaAttention] value_states shape: torch.Size([2, 3, 16, 64])
# [LlamaAttention] query_states after RoPE shape: torch.Size([2, 9, 16, 64])
# [LlamaAttention] key_states after RoPE shape: torch.Size([2, 3, 16, 64])
# [eager_attention_forward] key_states shape: torch.Size([2, 9, 16, 64])
# [eager_attention_forward] value_states shape: torch.Size([2, 9, 16, 64])
# [eager_attention_forward] attn_weights shape before mask: torch.Size([2, 9, 16, 16])
# [eager_attention_forward] attn_output shape before transpose: torch.Size([2, 9, 16, 64])
# [eager_attention_forward] attn_output shape after transpose: torch.Size([2, 16, 9, 64])
# [LlamaAttention] attn_output shape: torch.Size([2, 16, 9, 64])
# [LlamaAttention] attn_weights shape: torch.Size([2, 9, 16, 16])
# [LlamaAttention] attn_output after reshape shape: torch.Size([2, 16, 576])
# [LlamaAttention] attn_output after o_proj shape: torch.Size([2, 16, 576])
# output shape: torch.Size([2, 16, 576])
# attn_weights shape: torch.Size([2, 9, 16, 16])