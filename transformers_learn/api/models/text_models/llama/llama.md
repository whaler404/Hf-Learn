# HuggingFaceTB/SmolLM2-135M

## model config
```json
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "hidden_act": "silu",
  "hidden_size": 576,
  "initializer_range": 0.041666666666666664,
  "intermediate_size": 1536,
  "is_llama_config": true,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 9,
  "num_hidden_layers": 30,
  "num_key_value_heads": 3,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_interleaved": false,
  "rope_scaling": null,
  "rope_theta": 100000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": true,
  "vocab_size": 49152
}
```

## model print
```python
LlamaForQuestionAnswering(
  (transformer): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (qa_outputs): Linear(in_features=576, out_features=2, bias=True)
)
```

## torch info
```bash
======================================================================
Layer (type:depth-idx)                        Param #
======================================================================
LlamaForQuestionAnswering                     --
├─LlamaModel: 1-1                             --
│    └─Embedding: 2-1                         28,311,552
│    └─ModuleList: 2-2                        --
│    │    └─LlamaDecoderLayer: 3-1            3,540,096
│    │    └─LlamaDecoderLayer: 3-2            3,540,096
│    │    └─LlamaDecoderLayer: 3-3            3,540,096
│    │    └─LlamaDecoderLayer: 3-4            3,540,096
│    │    └─LlamaDecoderLayer: 3-5            3,540,096
│    │    └─LlamaDecoderLayer: 3-6            3,540,096
│    │    └─LlamaDecoderLayer: 3-7            3,540,096
│    │    └─LlamaDecoderLayer: 3-8            3,540,096
│    │    └─LlamaDecoderLayer: 3-9            3,540,096
│    │    └─LlamaDecoderLayer: 3-10           3,540,096
│    │    └─LlamaDecoderLayer: 3-11           3,540,096
│    │    └─LlamaDecoderLayer: 3-12           3,540,096
│    │    └─LlamaDecoderLayer: 3-13           3,540,096
│    │    └─LlamaDecoderLayer: 3-14           3,540,096
│    │    └─LlamaDecoderLayer: 3-15           3,540,096
│    │    └─LlamaDecoderLayer: 3-16           3,540,096
│    │    └─LlamaDecoderLayer: 3-17           3,540,096
│    │    └─LlamaDecoderLayer: 3-18           3,540,096
│    │    └─LlamaDecoderLayer: 3-19           3,540,096
│    │    └─LlamaDecoderLayer: 3-20           3,540,096
│    │    └─LlamaDecoderLayer: 3-21           3,540,096
│    │    └─LlamaDecoderLayer: 3-22           3,540,096
│    │    └─LlamaDecoderLayer: 3-23           3,540,096
│    │    └─LlamaDecoderLayer: 3-24           3,540,096
│    │    └─LlamaDecoderLayer: 3-25           3,540,096
│    │    └─LlamaDecoderLayer: 3-26           3,540,096
│    │    └─LlamaDecoderLayer: 3-27           3,540,096
│    │    └─LlamaDecoderLayer: 3-28           3,540,096
│    │    └─LlamaDecoderLayer: 3-29           3,540,096
│    │    └─LlamaDecoderLayer: 3-30           3,540,096
│    └─LlamaRMSNorm: 2-3                      576
│    └─LlamaRotaryEmbedding: 2-4              --
├─Linear: 1-2                                 1,154
======================================================================
Total params: 134,516,162
Trainable params: 134,516,162
Non-trainable params: 0
======================================================================
```

## 源码阅读

### GQA
一般的分组自注意力机制，qkv 为[b,h,l,d]，只在组内做注意力机制，得到 [b,h,l,l] 最后得到 [b,l,h*d] 。但是 llama 的分组注意力head 分为 q_head 和 kv_head ，先得到 [b,qh,l,d] 和 [b,kvh,l,d] ，由于 kv_groups_num * kv_head = q_head， 对 [b,kvh,l,d] copy 几份得到 [b,qh,l,d]，再进行一般的分组注意力惩罚。这样可能是减少 kv 投影层的参数量，不同的 query head 也能从相同的 kv head 中学到东西

### RoPE

另一个是 rope，有点类似于绝对位置编码，都用了三角函数，不过 rope 把三角函数的位置信息和 qk 相乘而不是相加，将特征维度两两分组，初始化 pos_ids 为 [len,] ，以及 inv_freqs 为 [len//2,] ，二者做外积 pos_ids@inv_freqs 得到 [len, len//2] ，然后 copy 一遍得到 freqs 为 [len, len] ，得到 cos 和 sin ，然后对 qk 的后半部分做 roteta_half ，即 `q_embed = (q * cos) + (rotate_half(q) * sin)` ，得到嵌入位置信息的位置编码
```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
return q_embed, k_embed
```
[rope 公式详解](https://www.bilibili.com/video/BV1Mj421R7JQ)<br>
[知乎 rope 详解](https://www.zhihu.com/tardis/zm/art/647109286)<br>

### SwiGLU

激活函数使用了 SwiGLU，就是 Swish 加上门控网络，在 LlamaMLP 中实现
[详解SwiGLU激活函数](https://zhuanlan.zhihu.com/p/31289994147)<br>

### RMSNorm

每层输出使用了 RMSNorm 归一化，区别于层归一化，舍弃了中心化，直接用方差作为缩放因子
[RMSNorm 详解](https://zhuanlan.zhihu.com/p/669071548)<br>

### mask

自回归生成，使用 `torch.triu` 生成负无穷的上三角矩阵掩码，把看不见的地方用负无穷遮盖掉，所以掩码操作是加法而不是乘法。
对于使用 kv cache 做增量推理，cache position 关注新生成的窗口内的 token ，其他部分看不到。kv cache 是对 key 和 value 投影后的结果 cache，每次自回归推理一个 token，只需要重新生成新的当前位置的 qkv，然后 cat 到 cache 中，
如果有 padding mask，将前面 padding 的位置填充负无穷掩盖掉，自回归掩码看不到前面的 padding 填充，通过 padding mask 和 casual mask 在前 mask_length 做异或运算更新 casual mask 。

1. 如果传入的 attention_mask 已经是 4D（如 [batch, 1, query_len, key_len]），直接用。
2. 否则，先生成一个全负无穷（min_dtype）的上三角矩阵（triu），保证只能看到自己和之前的 token。
3. 如果有 padding mask，则将 padding 位置也 mask 掉（即填充为负无穷）。
4. 最终输出 shape 为 [batch_size, 1, query_length, key_value_length]，可直接用于 attention 权重加法。

cache_position 主要用于增量推理（如生成时一步步 append token），它表示当前 batch 内每个序列“新 token”在全序列中的实际位置。
保证 causal mask 只允许每个 token 看到自己和之前的 token，增量推理时 mask 只对新 token 有效，历史 token 不再重复 mask。
padding mask 保证 attention 不会 attend 到 padding（填充）位置。

[kv cache 原理讲解](https://zhuanlan.zhihu.com/p/662498827)<br>

[LLaMA代码解读——Huggingface版](https://zhuanlan.zhihu.com/p/679696511)<br>

### generate

自回归生成的过程中，有一个不断生成的 loop ，比如当前到 t 步，输入为 [1,t] ，输出为 [1,t,d] ，贪婪生成从最后一个输出 [1,1,d] 得到 next token ，拼接到序列上得到 [1,t+1] ，进入下一个 loop 中，最后输出 input ids 和多轮生成的 tokens 的拼接