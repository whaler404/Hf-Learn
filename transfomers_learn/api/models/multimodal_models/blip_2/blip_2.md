# Salesforce/blip2-opt-2.7b

## model config
```json
{
  "architectures": [
    "Blip2ForConditionalGeneration"
  ],
  "image_text_hidden_size": 256,
  "image_token_index": 50265,
  "initializer_factor": 1.0,
  "initializer_range": 0.02,
  "model_type": "blip-2",
  "num_query_tokens": 32,
  "qformer_config": {
    "classifier_dropout": null,
    "model_type": "blip_2_qformer"
  },
  "text_config": {
    "_name_or_path": "facebook/opt-2.7b",
    "activation_dropout": 0.0,
    "architectures": [
      "OPTForCausalLM"
    ],
    "eos_token_id": 50118,
    "ffn_dim": 10240,
    "hidden_size": 2560,
    "model_type": "opt",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "prefix": "</s>",
    "torch_dtype": "float16",
    "vocab_size": 50304,
    "word_embed_proj_dim": 2560
  },
  "torch_dtype": "float32",
  "transformers_version": "4.47.0.dev0",
  "use_decoder_only_language_model": true,
  "vision_config": {
    "dropout": 0.0,
    "initializer_factor": 1.0,
    "model_type": "blip_2_vision_model",
    "num_channels": 3,
    "projection_dim": 512
  }
}
```

## model print
```python
Blip2Model(
  (vision_model): Blip2VisionModel(
    (embeddings): Blip2VisionEmbeddings(
      (patch_embedding): Conv2d(3, 1408, kernel_size=(14, 14), stride=(14, 14))
    )
    (encoder): Blip2Encoder(
      (layers): ModuleList(
        (0-2): 3 x Blip2EncoderLayer(
          (self_attn): Blip2Attention(
            (qkv): Linear(in_features=1408, out_features=4224, bias=True)
            (projection): Linear(in_features=1408, out_features=1408, bias=True)
          )
          (layer_norm1): LayerNorm((1408,), eps=1e-06, elementwise_affine=True)
          (mlp): Blip2MLP(
            (activation_fn): GELUActivation()
            (fc1): Linear(in_features=1408, out_features=6144, bias=True)
            (fc2): Linear(in_features=6144, out_features=1408, bias=True)
          )
          (layer_norm2): LayerNorm((1408,), eps=1e-06, elementwise_affine=True)
        )
      )
    )
    (post_layernorm): LayerNorm((1408,), eps=1e-06, elementwise_affine=True)
  )
  (qformer): Blip2QFormerModel(
    (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (encoder): Blip2QFormerEncoder(
      (layer): ModuleList(
        (0): Blip2QFormerLayer(
          (attention): Blip2QFormerAttention(
            (attention): Blip2QFormerMultiHeadAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): Blip2QFormerSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (crossattention): Blip2QFormerAttention(
            (attention): Blip2QFormerMultiHeadAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=1408, out_features=768, bias=True)
              (value): Linear(in_features=1408, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): Blip2QFormerSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate_query): Blip2QFormerIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output_query): Blip2QFormerOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (1): Blip2QFormerLayer(
          (attention): Blip2QFormerAttention(
            (attention): Blip2QFormerMultiHeadAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): Blip2QFormerSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate_query): Blip2QFormerIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output_query): Blip2QFormerOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (language_projection): Linear(in_features=768, out_features=2560, bias=True)
  (language_model): OPTForCausalLM(
    (model): OPTModel(
      (decoder): OPTDecoder(
        (embed_tokens): Embedding(50304, 2560, padding_idx=1)
        (embed_positions): OPTLearnedPositionalEmbedding(2050, 2560)
        (final_layer_norm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (layers): ModuleList(
          (0-2): 3 x OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): Linear(in_features=2560, out_features=2560, bias=True)
              (v_proj): Linear(in_features=2560, out_features=2560, bias=True)
              (q_proj): Linear(in_features=2560, out_features=2560, bias=True)
              (out_proj): Linear(in_features=2560, out_features=2560, bias=True)
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=2560, out_features=10240, bias=True)
            (fc2): Linear(in_features=10240, out_features=2560, bias=True)
            (final_layer_norm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
    )
    (lm_head): Linear(in_features=2560, out_features=50304, bias=False)
  )
)
```

## torch info
```bash
===============================================================================================
Layer (type:depth-idx)                                                 Param #
===============================================================================================
Blip2Model                                                             24,576
├─Blip2VisionModel: 1-1                                                --
│    └─Blip2VisionEmbeddings: 2-1                                      363,264
│    │    └─Conv2d: 3-1                                                829,312
│    └─Blip2Encoder: 2-2                                               --
│    │    └─ModuleList: 3-2                                            75,750,528
│    └─LayerNorm: 2-3                                                  2,816
├─Blip2QFormerModel: 1-2                                               --
│    └─LayerNorm: 2-4                                                  1,536
│    └─Dropout: 2-5                                                    --
│    └─Blip2QFormerEncoder: 2-6                                        --
│    │    └─ModuleList: 3-3                                            17,522,688
├─Linear: 1-3                                                          1,968,640
├─OPTForCausalLM: 1-4                                                  --
│    └─OPTModel: 2-7                                                   --
│    │    └─OPTDecoder: 3-4                                            370,060,800
│    └─Linear: 2-8                                                     128,778,240
===============================================================================================
Total params: 595,302,400
Trainable params: 595,302,400
Non-trainable params: 0
===============================================================================================
```

## 源码阅读

blip2 增加了个 qformer 来对齐大模型和视觉模型，通过表示学习和生成学习两个阶段来对齐模型。表示学习阶段，冻结视觉模型，训练 qformer，使用 learned query 和 text 的不同掩码方式来控制 image 和 text transformer 的交互方式，单模态注意力掩码让 query 和 text 专注于自身，避免信息泄露，自回归掩码只让 text 关注前面的 query，双向注意力掩码让所有的 query 和 text 都能相互关注；生成学习阶段，冻结 qformer 和 llm ，通过全连接层将 qformer 输出的 query 进行投影，连接到输入文本嵌入前面

[一文读懂BLIP和BLIP-2多模态预训练](https://zhuanlan.zhihu.com/p/640887802)<br>