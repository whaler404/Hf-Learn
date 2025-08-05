# Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4

## model config
```json
{
  "architectures": [
    "Qwen2MoeForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "decoder_sparse_step": 1,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 5632,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2_moe",
  "moe_intermediate_size": 1408,
  "norm_topk_prob": false,
  "num_attention_heads": 16,
  "num_experts": 60,
  "num_experts_per_tok": 4,
  "num_hidden_layers": 24,
  "num_key_value_heads": 16,
  "output_router_logits": false,
  "quantization_config": {
    "batch_size": 1,
    "bits": 4,
    "block_name_to_quantize": null,
    "cache_block_outputs": true,
    "damp_percent": 0.01,
    "dataset": null,
    "desc_act": false,
    "exllama_config": {
      "version": 2
    },
    "group_size": 128,
    "max_input_length": null,
    "model_seqlen": null,
    "module_name_preceding_first_block": null,
    "modules_in_block_to_quantize": [
      [
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj"
      ],
      [
        "self_attn.o_proj"
      ],
      [
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.gate_proj"
      ],
      [
        "mlp.shared_expert.down_proj"
      ],
      [
        "mlp.experts.0.up_proj",
        "mlp.experts.1.up_proj",
        "mlp.experts.2.up_proj",
        ...
        "mlp.experts.57.gate_proj",
        "mlp.experts.58.gate_proj",
        "mlp.experts.59.gate_proj"
      ],
      [
        "mlp.experts.0.down_proj",
        "mlp.experts.1.down_proj",
        "mlp.experts.2.down_proj",
        ...
        "mlp.experts.57.down_proj",
        "mlp.experts.58.down_proj",
        "mlp.experts.59.down_proj"
      ]
    ],
    "pad_token_id": null,
    "quant_method": "gptq",
    "sym": true,
    "tokenizer": null,
    "true_sequential": true,
    "use_cuda_fp16": false,
    "use_exllama": true
  },
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "router_aux_loss_coef": 0.001,
  "shared_expert_intermediate_size": 5632,
  "sliding_window": 32768,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.39.0.dev0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

## model print
```python
Qwen2MoeForCausalLM(
  (model): Qwen2MoeModel(
    (embed_tokens): Embedding(1519, 2048)
    (layers): ModuleList(
      (0-2): 3 x Qwen2MoeDecoderLayer(
        (self_attn): Qwen2MoeSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): Qwen2MoeRotaryEmbedding()
        )
        (mlp): Qwen2MoeSparseMoeBlock(
          (gate): Linear(in_features=2048, out_features=4, bias=False)
          (experts): ModuleList(
            (0-3): 4 x Qwen2MoeMLP(
              (gate_proj): Linear(in_features=2048, out_features=1408, bias=False)
              (up_proj): Linear(in_features=2048, out_features=1408, bias=False)
              (down_proj): Linear(in_features=1408, out_features=2048, bias=False)
              (act_fn): SiLU()
            )
          )
          (shared_expert): Qwen2MoeMLP(
            (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
            (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
            (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
            (act_fn): SiLU()
          )
          (shared_expert_gate): Linear(in_features=2048, out_features=1, bias=False)
        )
        (input_layernorm): Qwen2MoeRMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen2MoeRMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen2MoeRMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen2MoeRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=1519, bias=False)
)
```

## torch info
```bash
================================================================================
Layer (type:depth-idx)                                  Param #
================================================================================
Qwen2MoeForCausalLM                                     --
├─Qwen2MoeModel: 1-1                                    --
│    └─Embedding: 2-1                                   3,110,912
│    └─ModuleList: 2-2                                  --
│    │    └─Qwen2MoeDecoderLayer: 3-1                   86,003,712
│    │    └─Qwen2MoeDecoderLayer: 3-2                   86,003,712
│    │    └─Qwen2MoeDecoderLayer: 3-3                   86,003,712
│    └─Qwen2MoeRMSNorm: 2-3                             2,048
│    └─Qwen2MoeRotaryEmbedding: 2-4                     --
├─Linear: 1-2                                           3,110,912
================================================================================
Total params: 264,235,008
Trainable params: 264,235,008
Non-trainable params: 0
================================================================================
```

## 源码阅读

qwen1.5 moe 的混合专家模块，采用了 gate 网络，输出各个 expert 的 logits ，使用 torch.topk 挑选并用独热码构造 mask ，（ps 在 pytorch 中， topk 和 embedding 操作是可导的）

`router_logits = self.gate(hidden_states)`：每个 token 得到每个专家的分数
`routing_weights = softmax(router_logits)`：归一化为概率
`routing_weights, selected_experts = topk(routing_weights, self.top_k)`：每个 token 选 top-k 概率最大的专家
如果 `norm_topk_prob`，则对 top-k 概率再归一化
`routing_weights` 转回原始 dtype

[huggingface moe 讲解](https://zhuanlan.zhihu.com/p/674698482)<br>