# Salesforce/blip-image-captioning-base

## model config
```json
{
  "_commit_hash": null,
  "architectures": [
    "BlipForConditionalGeneration"
  ],
  "image_text_hidden_size": 256,
  "initializer_factor": 1.0,
  "logit_scale_init_value": 2.6592,
  "model_type": "blip",
  "projection_dim": 512,
  "text_config": {
    "_name_or_path": "",
    "add_cross_attention": false,
    "architectures": null,
    "attention_probs_dropout_prob": 0.0,
    "bad_words_ids": null,
    "begin_suppress_tokens": null,
    "bos_token_id": 30522,
    "chunk_size_feed_forward": 0,
    "cross_attention_hidden_size": null,
    "decoder_start_token_id": null,
    "diversity_penalty": 0.0,
    "do_sample": false,
    "early_stopping": false,
    "encoder_no_repeat_ngram_size": 0,
    "eos_token_id": 2,
    "exponential_decay_length_penalty": null,
    "finetuning_task": null,
    "forced_bos_token_id": null,
    "forced_eos_token_id": null,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 768,
    "id2label": {
      "0": "LABEL_0",
      "1": "LABEL_1"
    },
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "is_decoder": true,
    "is_encoder_decoder": false,
    "label2id": {
      "LABEL_0": 0,
      "LABEL_1": 1
    },
    "layer_norm_eps": 1e-12,
    "length_penalty": 1.0,
    "max_length": 20,
    "max_position_embeddings": 512,
    "min_length": 0,
    "model_type": "blip_text_model",
    "no_repeat_ngram_size": 0,
    "num_attention_heads": 12,
    "num_beam_groups": 1,
    "num_beams": 1,
    "num_hidden_layers": 12,
    "num_return_sequences": 1,
    "output_attentions": false,
    "output_hidden_states": false,
    "output_scores": false,
    "pad_token_id": 0,
    "prefix": null,
    "problem_type": null,
    "projection_dim": 768,
    "pruned_heads": {},
    "remove_invalid_values": false,
    "repetition_penalty": 1.0,
    "return_dict": true,
    "return_dict_in_generate": false,
    "sep_token_id": 102,
    "suppress_tokens": null,
    "task_specific_params": null,
    "temperature": 1.0,
    "tf_legacy_loss": false,
    "tie_encoder_decoder": false,
    "tie_word_embeddings": true,
    "tokenizer_class": null,
    "top_k": 50,
    "top_p": 1.0,
    "torch_dtype": null,
    "torchscript": false,
    "transformers_version": "4.26.0.dev0",
    "typical_p": 1.0,
    "use_bfloat16": false,
    "use_cache": true,
    "vocab_size": 30524
  },
  "torch_dtype": "float32",
  "transformers_version": null,
  "vision_config": {
    "_name_or_path": "",
    "add_cross_attention": false,
    "architectures": null,
    "attention_dropout": 0.0,
    "bad_words_ids": null,
    "begin_suppress_tokens": null,
    "bos_token_id": null,
    "chunk_size_feed_forward": 0,
    "cross_attention_hidden_size": null,
    "decoder_start_token_id": null,
    "diversity_penalty": 0.0,
    "do_sample": false,
    "dropout": 0.0,
    "early_stopping": false,
    "encoder_no_repeat_ngram_size": 0,
    "eos_token_id": null,
    "exponential_decay_length_penalty": null,
    "finetuning_task": null,
    "forced_bos_token_id": null,
    "forced_eos_token_id": null,
    "hidden_act": "gelu",
    "hidden_size": 768,
    "id2label": {
      "0": "LABEL_0",
      "1": "LABEL_1"
    },
    "image_size": 384,
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "is_decoder": false,
    "is_encoder_decoder": false,
    "label2id": {
      "LABEL_0": 0,
      "LABEL_1": 1
    },
    "layer_norm_eps": 1e-05,
    "length_penalty": 1.0,
    "max_length": 20,
    "min_length": 0,
    "model_type": "blip_vision_model",
    "no_repeat_ngram_size": 0,
    "num_attention_heads": 12,
    "num_beam_groups": 1,
    "num_beams": 1,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "num_return_sequences": 1,
    "output_attentions": false,
    "output_hidden_states": false,
    "output_scores": false,
    "pad_token_id": null,
    "patch_size": 16,
    "prefix": null,
    "problem_type": null,
    "projection_dim": 512,
    "pruned_heads": {},
    "remove_invalid_values": false,
    "repetition_penalty": 1.0,
    "return_dict": true,
    "return_dict_in_generate": false,
    "sep_token_id": null,
    "suppress_tokens": null,
    "task_specific_params": null,
    "temperature": 1.0,
    "tf_legacy_loss": false,
    "tie_encoder_decoder": false,
    "tie_word_embeddings": true,
    "tokenizer_class": null,
    "top_k": 50,
    "top_p": 1.0,
    "torch_dtype": null,
    "torchscript": false,
    "transformers_version": "4.26.0.dev0",
    "typical_p": 1.0,
    "use_bfloat16": false
  }
}
```

## model print
```python
BlipModel(
  (text_model): BlipTextModel(
    (embeddings): BlipTextEmbeddings(
      (word_embeddings): Embedding(30524, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (encoder): BlipTextEncoder(
      (layer): ModuleList(
        (0-1): 2 x BlipTextLayer(
          (attention): BlipTextAttention(
            (self): BlipTextSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): BlipTextSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (crossattention): BlipTextAttention(
            (self): BlipTextSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): BlipTextSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): BlipTextIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BlipTextOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
      )
    )
    (pooler): BlipTextPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (vision_model): BlipVisionModel(
    (embeddings): BlipVisionEmbeddings(
      (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    )
    (encoder): BlipEncoder(
      (layers): ModuleList(
        (0-1): 2 x BlipEncoderLayer(
          (self_attn): BlipAttention(
            (dropout): Dropout(p=0.0, inplace=False)
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (projection): Linear(in_features=768, out_features=768, bias=True)
          )
          (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): BlipMLP(
            (activation_fn): GELUActivation()
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
          )
          (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (post_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (visual_projection): Linear(in_features=768, out_features=512, bias=False)
  (text_projection): Linear(in_features=768, out_features=512, bias=False)
)
```

## torch info
```bash
=====================================================================================
Layer (type:depth-idx)                                       Param #
=====================================================================================
BlipModel                                                    1
├─BlipTextModel: 1-1                                         --
│    └─BlipTextEmbeddings: 2-1                               --
│    │    └─Embedding: 3-1                                   23,442,432
│    │    └─Embedding: 3-2                                   393,216
│    │    └─LayerNorm: 3-3                                   1,536
│    │    └─Dropout: 3-4                                     --
│    └─BlipTextEncoder: 2-2                                  --
│    │    └─ModuleList: 3-5                                  18,903,552
│    └─BlipTextPooler: 2-3                                   --
│    │    └─Linear: 3-6                                      590,592
│    │    └─Tanh: 3-7                                        --
├─BlipVisionModel: 1-2                                       --
│    └─BlipVisionEmbeddings: 2-4                             443,904
│    │    └─Conv2d: 3-8                                      590,592
│    └─BlipEncoder: 2-5                                      --
│    │    └─ModuleList: 3-9                                  14,175,744
│    └─LayerNorm: 2-6                                        1,536
├─Linear: 1-3                                                393,216
├─Linear: 1-4                                                393,216
=====================================================================================
Total params: 59,329,537
Trainable params: 59,329,537
Non-trainable params: 0
=====================================================================================
```

## 源码阅读

看 blip ，其实和 clip 还是有部分区别的，都是语言图像模型，blip 分阶段的训练模型，每阶段的训练目标也不一样，第一阶段训练图像文本对的对比损失，第二阶段训练图像文本对的匹配损失的文本 encoder，第三阶段训练自回归生成标题的文本 decoder。boostrap 的过程就是 captioner 生成的和 filter 过滤后的图像文本对，结合人工构造的，作为新的数据集重新训练模型。

[一文读懂BLIP和BLIP-2多模态预训练](https://zhuanlan.zhihu.com/p/640887802)<br>