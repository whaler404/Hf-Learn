# Qwen/Qwen2.5-0.5B

## model config
```json
{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 24,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": true,
  "use_mrope": false,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

## model print
```python
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 896)
    (layers): ModuleList(
      (0-23): 24 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=896, out_features=896, bias=True)
          (k_proj): Linear(in_features=896, out_features=128, bias=True)
          (v_proj): Linear(in_features=896, out_features=128, bias=True)
          (o_proj): Linear(in_features=896, out_features=896, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
          (up_proj): Linear(in_features=896, out_features=4864, bias=False)
          (down_proj): Linear(in_features=4864, out_features=896, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((896,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=896, out_features=151936, bias=False)
)
```

## torch info
```bash
======================================================================
Layer (type:depth-idx)                        Param #
======================================================================
Qwen2ForCausalLM                              --
├─Qwen2Model: 1-1                             --
│    └─Embedding: 2-1                         136,134,656
│    └─ModuleList: 2-2                        --
│    │    └─Qwen2DecoderLayer: 3-1            14,912,384
│    │    └─Qwen2DecoderLayer: 3-2            14,912,384
│    │    └─Qwen2DecoderLayer: 3-3            14,912,384
│    │    └─Qwen2DecoderLayer: 3-4            14,912,384
│    │    └─Qwen2DecoderLayer: 3-5            14,912,384
│    │    └─Qwen2DecoderLayer: 3-6            14,912,384
│    │    └─Qwen2DecoderLayer: 3-7            14,912,384
│    │    └─Qwen2DecoderLayer: 3-8            14,912,384
│    │    └─Qwen2DecoderLayer: 3-9            14,912,384
│    │    └─Qwen2DecoderLayer: 3-10           14,912,384
│    │    └─Qwen2DecoderLayer: 3-11           14,912,384
│    │    └─Qwen2DecoderLayer: 3-12           14,912,384
│    │    └─Qwen2DecoderLayer: 3-13           14,912,384
│    │    └─Qwen2DecoderLayer: 3-14           14,912,384
│    │    └─Qwen2DecoderLayer: 3-15           14,912,384
│    │    └─Qwen2DecoderLayer: 3-16           14,912,384
│    │    └─Qwen2DecoderLayer: 3-17           14,912,384
│    │    └─Qwen2DecoderLayer: 3-18           14,912,384
│    │    └─Qwen2DecoderLayer: 3-19           14,912,384
│    │    └─Qwen2DecoderLayer: 3-20           14,912,384
│    │    └─Qwen2DecoderLayer: 3-21           14,912,384
│    │    └─Qwen2DecoderLayer: 3-22           14,912,384
│    │    └─Qwen2DecoderLayer: 3-23           14,912,384
│    │    └─Qwen2DecoderLayer: 3-24           14,912,384
│    └─Qwen2RMSNorm: 2-3                      896
│    └─Qwen2RotaryEmbedding: 2-4              --
├─Linear: 1-2                                 136,134,656
======================================================================
Total params: 630,167,424
Trainable params: 630,167,424
Non-trainable params: 0
======================================================================
```