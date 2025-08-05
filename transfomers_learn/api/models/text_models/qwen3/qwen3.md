# Qwen/Qwen3-0.6B

## model config
```json
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

## model print
```python
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(1519, 1024)
    (layers): ModuleList(
      (0-2): 3 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=1519, bias=False)
)
```

## torch info
```bash
======================================================================
Layer (type:depth-idx)                        Param #
======================================================================
Qwen3ForCausalLM                              --
├─Qwen3Model: 1-1                             --
│    └─Embedding: 2-1                         1,555,456
│    └─ModuleList: 2-2                        --
│    │    └─Qwen3DecoderLayer: 3-1            15,730,944
│    │    └─Qwen3DecoderLayer: 3-2            15,730,944
│    │    └─Qwen3DecoderLayer: 3-3            15,730,944
│    └─Qwen3RMSNorm: 2-3                      1,024
│    └─Qwen3RotaryEmbedding: 2-4              --
├─Linear: 1-2                                 1,555,456
======================================================================
Total params: 50,304,768
Trainable params: 50,304,768
Non-trainable params: 0
======================================================================
```

## 源码阅读
