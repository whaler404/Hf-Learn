# Qwen/Qwen2.5-VL-3B-Instruct

## model config
```json
{
  "architectures": [
    "Qwen2_5_VLForConditionalGeneration"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "vision_start_token_id": 151652,
  "vision_end_token_id": 151653,
  "vision_token_id": 151654,
  "image_token_id": 151655,
  "video_token_id": 151656,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 128000,
  "max_window_layers": 70,
  "model_type": "qwen2_5_vl",
  "num_attention_heads": 16,
  "num_hidden_layers": 36,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.41.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vision_config": {
    "depth": 32,
    "hidden_act": "silu",
    "hidden_size": 1280,
    "intermediate_size": 3420,
    "num_heads": 16,
    "in_chans": 3,
    "out_hidden_size": 2048,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "window_size": 112,
    "fullatt_block_indexes": [
      7,
      15,
      23,
      31
    ],
    "tokens_per_second": 2,
    "temporal_patch_size": 2
  },
  "rope_scaling": {
    "type": "mrope",
    "mrope_section": [
      16,
      24,
      24
    ]
  },
  "vocab_size": 151936
}
```

## model print
```python
Qwen2_5_VLModel(
  (visual): Qwen2_5_VisionTransformerPretrainedModel(
    (patch_embed): Qwen2_5_VisionPatchEmbed(
      (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
    )
    (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
    (blocks): ModuleList(
      (0-2): 3 x Qwen2_5_VLVisionBlock(
        (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
        (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
        (attn): Qwen2_5_VLVisionSdpaAttention(
          (qkv): Linear(in_features=1280, out_features=3840, bias=True)
          (proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (mlp): Qwen2_5_VLMLP(
          (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
          (act_fn): SiLU()
        )
      )
    )
    (merger): Qwen2_5_VLPatchMerger(
      (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
      (mlp): Sequential(
        (0): Linear(in_features=5120, out_features=5120, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=5120, out_features=2048, bias=True)
      )
    )
  )
  (language_model): Qwen2_5_VLTextModel(
    (embed_tokens): Embedding(1519, 2048)
    (layers): ModuleList(
      (0-2): 3 x Qwen2_5_VLDecoderLayer(
        (self_attn): Qwen2_5_VLSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=256, bias=True)
          (v_proj): Linear(in_features=2048, out_features=256, bias=True)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): Qwen2_5_VLRotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (up_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen2_5_VLRotaryEmbedding()
  )
)
```

## torch info
```bash
=====================================================================================
Layer (type:depth-idx)                                       Param #
=====================================================================================
Qwen2_5_VLModel                                              --
├─Qwen2_5_VisionTransformerPretrainedModel: 1-1              --
│    └─Qwen2_5_VisionPatchEmbed: 2-1                         --
│    │    └─Conv3d: 3-1                                      1,505,280
│    └─Qwen2_5_VisionRotaryEmbedding: 2-2                    --
│    └─ModuleList: 2-3                                       --
│    │    └─Qwen2_5_VLVisionBlock: 3-2                       19,702,200
│    │    └─Qwen2_5_VLVisionBlock: 3-3                       19,702,200
│    │    └─Qwen2_5_VLVisionBlock: 3-4                       19,702,200
│    └─Qwen2_5_VLPatchMerger: 2-4                            --
│    │    └─Qwen2RMSNorm: 3-5                                1,280
│    │    └─Sequential: 3-6                                  36,707,328
├─Qwen2_5_VLTextModel: 1-2                                   --
│    └─Embedding: 2-5                                        3,110,912
│    └─ModuleList: 2-6                                       --
│    │    └─Qwen2_5_VLDecoderLayer: 3-7                      77,076,992
│    │    └─Qwen2_5_VLDecoderLayer: 3-8                      77,076,992
│    │    └─Qwen2_5_VLDecoderLayer: 3-9                      77,076,992
│    └─Qwen2RMSNorm: 2-7                                     2,048
│    └─Qwen2_5_VLRotaryEmbedding: 2-8                        --
=====================================================================================
```