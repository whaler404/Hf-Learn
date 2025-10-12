# google-bert/bert-base-cased

## model config
```json
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.6.0.dev0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 28996
}
```

## model print
```python
BertForMaskedLM(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(28996, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
  )
  (cls): BertOnlyMLMHead(
    (predictions): BertLMPredictionHead(
      (transform): BertPredictionHeadTransform(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (transform_act_fn): GELUActivation()
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      )
      (decoder): Linear(in_features=768, out_features=28996, bias=True)
    )
  )
)
```

## torch info
```bash
=====================================================================================
Layer (type:depth-idx)                                       Param #
=====================================================================================
BertForMaskedLM                                              --
├─BertModel: 1-1                                             --
│    └─BertEmbeddings: 2-1                                   --
│    │    └─Embedding: 3-1                                   22,268,928
│    │    └─Embedding: 3-2                                   393,216
│    │    └─Embedding: 3-3                                   1,536
│    │    └─LayerNorm: 3-4                                   1,536
│    │    └─Dropout: 3-5                                     --
│    └─BertEncoder: 2-2                                      --
│    │    └─ModuleList: 3-6                                  85,054,464
├─BertOnlyMLMHead: 1-2                                       --
│    └─BertLMPredictionHead: 2-3                             --
│    │    └─BertPredictionHeadTransform: 3-7                 592,128
│    │    └─Linear: 3-8                                      22,297,924
=====================================================================================
Total params: 130,609,732
Trainable params: 130,609,732
Non-trainable params: 0
=====================================================================================
```

## 源码阅读


1. cross-attention且有缓存时，直接用缓存
2. cross-attention无缓存时，用encoder_hidden_states生成key/value
3. 非cross-attention有缓存时，将新key/value和缓存拼接
4. 否则用hidden_states生成key/value

使用可学习的相对位置编码，得到 query 和 key 的相对位置，输入 embedding 层中，一开始 query 为 [b,h,l,d] 和 key 为 [b,r,d] ，点积注意力得到 [b,h,l,r] ， query 和 key 的相对位置编码 [l,] 和 [r,] 计算相对位置得到 [l,r] ，嵌入后得到 [l,r,d] ，将 qk 分别和位置编码点积得到相对位置分数 [b,h,l,r] ，注意力分数加上 query 的相对位置分数和 key 的相对位置分数得到新的注意力分数。

bert 的 embedding 分为三部分，包括 token、position、type vocab 的嵌入编码
bert 使用 layernorm ，激活函数使用 gleu ， bert layer 如果是解码器，使用交叉注意力，
bert encoder 在 train 和 eval 的模式不一样，训练模式增加了 gradient checkpoint 来减少训练显存占用
[torch.nn.Module.train() & torch.nn.Module.eval()](https://www.cnblogs.com/wupiao/articles/13287061.html)<br>

bert 中也用了 gradient checkpoint 技术，通过只保存正向传播激活值计算方式来减少显存占用，在 llama 中也用了
[训练策略--梯度检查点Gradient Checkpointing](https://zhuanlan.zhihu.com/p/689971296)<br>