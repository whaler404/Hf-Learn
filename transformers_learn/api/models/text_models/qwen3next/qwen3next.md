# Qwen/Qwen3-Next-80B-A3B-Instruct

## model config
```json
{
    "architectures": [
        "Qwen3NextForCausalLM"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 1516,
    "decoder_sparse_step": 1,
    "eos_token_id": 1516,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 5120,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "mlp_only_layers": [],
    "model_type": "qwen3_next",
    "moe_intermediate_size": 512,
    "norm_topk_prob": true,
    "num_attention_heads": 16,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "num_hidden_layers": 1,
    "num_key_value_heads": 2,
    "output_router_logits": false,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-06,
    "rope_scaling": null,
    "rope_theta": 10000000,
    "router_aux_loss_coef": 0.001,
    "shared_expert_intermediate_size": 512,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.57.0.dev0",
    "use_cache": true,
    "use_sliding_window": false,
    "vocab_size": 1519
}
```

## model print
```python
Qwen3NextForCausalLM(
  (model): Qwen3NextModel(
    (embed_tokens): Embedding(1519, 2048)
    (layers): ModuleList(
      (0): Qwen3NextDecoderLayer(
        (linear_attn): Qwen3NextGatedDeltaNet(
          (act): SiLUActivation()
          (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
          (in_proj_qkvz): Linear(in_features=2048, out_features=12288, bias=False)
          (in_proj_ba): Linear(in_features=2048, out_features=64, bias=False)
          (norm): Qwen3NextRMSNormGated()
          (out_proj): Linear(in_features=4096, out_features=2048, bias=False)
        )
        (mlp): Qwen3NextSparseMoeBlock(
          (gate): Linear(in_features=2048, out_features=512, bias=False)
          (experts): ModuleList(
            (0-511): 512 x Qwen3NextMLP(
              (gate_proj): Linear(in_features=2048, out_features=512, bias=False)
              (up_proj): Linear(in_features=2048, out_features=512, bias=False)
              (down_proj): Linear(in_features=512, out_features=2048, bias=False)
              (act_fn): SiLUActivation()
            )
          )
          (shared_expert): Qwen3NextMLP(
            (gate_proj): Linear(in_features=2048, out_features=512, bias=False)
            (up_proj): Linear(in_features=2048, out_features=512, bias=False)
            (down_proj): Linear(in_features=512, out_features=2048, bias=False)
            (act_fn): SiLUActivation()
          )
          (shared_expert_gate): Linear(in_features=2048, out_features=1, bias=False)
        )
        (input_layernorm): Qwen3NextRMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen3NextRMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen3NextRMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen3NextRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=1519, bias=False)
)
```

## torch info
```bash
=====================================================================================
Layer (type:depth-idx)                                       Param #
=====================================================================================
Qwen3NextForCausalLM                                         --
├─Qwen3NextModel: 1-1                                        --
│    └─Embedding: 2-1                                        3,110,912
│    └─ModuleList: 2-2                                       --
│    │    └─Qwen3NextDecoderLayer: 3-1                       1,648,531,648
│    └─Qwen3NextRMSNorm: 2-3                                 2,048
│    └─Qwen3NextRotaryEmbedding: 2-4                         --
├─Linear: 1-2                                                3,110,912
=====================================================================================
Total params: 1,654,755,520
Trainable params: 1,654,755,520
Non-trainable params: 0
=====================================================================================
```

## 源码阅读
