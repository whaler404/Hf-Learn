# deepseek-ai/DeepSeek-R1

## model config
```json
{
    "architectures": [
        "DeepseekV3ForCausalLM"
    ],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_deepseek.DeepseekV3Config",
        "AutoModel": "modeling_deepseek.DeepseekV3Model",
        "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
    },
    "bos_token_id": 0,
    "eos_token_id": 1,
    "ep_size": 1,
    "first_k_dense_replace": 3,
    "hidden_act": "silu",
    "hidden_size": 7168,
    "initializer_range": 0.02,
    "intermediate_size": 18432,
    "kv_lora_rank": 512,
    "max_position_embeddings": 163840,
    "model_type": "deepseek_v3",
    "moe_intermediate_size": 2048,
    "moe_layer_freq": 1,
    "n_group": 8,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": true,
    "num_attention_heads": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 61,
    "num_key_value_heads": 128,
    "num_nextn_predict_layers": 1,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [
            128,
            128
        ]
    },
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn"
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "tie_word_embeddings": false,
    "topk_group": 4,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.46.3",
    "use_cache": true,
    "v_head_dim": 128,
    "vocab_size": 129280
}
```

## model print
```python
DeepseekV3ForCausalLM(
  (model): DeepseekV3Model(
    (embed_tokens): Embedding(1292, 7168)
    (layers): ModuleList(
      (0-2): 3 x DeepseekV3DecoderLayer(
        (self_attn): DeepseekV3Attention(
          (q_a_proj): Linear(in_features=7168, out_features=1536, bias=False)
          (q_a_layernorm): DeepseekV3RMSNorm((1536,), eps=1e-06)
          (q_b_proj): Linear(in_features=1536, out_features=24576, bias=False)
          (kv_a_proj_with_mqa): Linear(in_features=7168, out_features=576, bias=False)
          (kv_a_layernorm): DeepseekV3RMSNorm((512,), eps=1e-06)
          (kv_b_proj): Linear(in_features=512, out_features=32768, bias=False)
          (o_proj): Linear(in_features=16384, out_features=7168, bias=False)
        )
        (mlp): DeepseekV3MLP(
          (gate_proj): Linear(in_features=7168, out_features=1843, bias=False)
          (up_proj): Linear(in_features=7168, out_features=1843, bias=False)
          (down_proj): Linear(in_features=1843, out_features=7168, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): DeepseekV3RMSNorm((7168,), eps=1e-06)
        (post_attention_layernorm): DeepseekV3RMSNorm((7168,), eps=1e-06)
      )
    )
    (norm): DeepseekV3RMSNorm((7168,), eps=1e-06)
    (rotary_emb): DeepseekV3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=7168, out_features=1292, bias=False)
)
```

## torch info
```bash
===========================================================================
Layer (type:depth-idx)                             Param #
===========================================================================
DeepseekV3ForCausalLM                              --
├─DeepseekV3Model: 1-1                             --
│    └─Embedding: 2-1                              9,261,056
│    └─ModuleList: 2-2                             --
│    │    └─DeepseekV3DecoderLayer: 3-1            226,753,536
│    │    └─DeepseekV3DecoderLayer: 3-2            226,753,536
│    │    └─DeepseekV3DecoderLayer: 3-3            226,753,536
│    └─DeepseekV3RMSNorm: 2-3                      7,168
│    └─DeepseekV3RotaryEmbedding: 2-4              --
├─Linear: 1-2                                      9,261,056
===========================================================================
Total params: 698,789,888
Trainable params: 698,789,888
Non-trainable params: 0
===========================================================================
```