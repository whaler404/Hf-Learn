# qwen2-tiny-lora

## model summary

```bash
trainable params: 6,784 || all params: 13,508,304 || trainable%: 0.0502
=====================================================================================
Layer (type:depth-idx)                                       Param #
=====================================================================================
PeftModelForSequenceClassification                           --
├─LoraModel: 1-1                                             --
│    └─Qwen2ForSequenceClassification: 2-1                   --
│    │    └─Qwen2Model: 3-1                                  13,508,048
│    │    └─ModulesToSaveWrapper: 3-2                        256
=====================================================================================
Total params: 13,508,304
Trainable params: 6,784
Non-trainable params: 13,501,520
=====================================================================================
```

## model print

```python
PeftModelForSequenceClassification(
  (base_model): LoraModel(
    (model): Qwen2ForSequenceClassification(
      (model): Qwen2Model(
        (embed_tokens): Embedding(151936, 64)
        (layers): ModuleList(
          (0-3): 4 x Qwen2DecoderLayer(
            (self_attn): Qwen2Attention(
              (q_proj): lora.Linear(
                (base_layer): Linear(in_features=64, out_features=60, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=64, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=60, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (k_proj): Linear(in_features=64, out_features=20, bias=True)
              (v_proj): lora.Linear(
                (base_layer): Linear(in_features=64, out_features=20, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=64, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=20, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (o_proj): Linear(in_features=60, out_features=64, bias=False)
            )
            (mlp): Qwen2MLP(
              (gate_proj): Linear(in_features=64, out_features=4864, bias=False)
              (up_proj): Linear(in_features=64, out_features=4864, bias=False)
              (down_proj): Linear(in_features=4864, out_features=64, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen2RMSNorm((64,), eps=1e-06)
            (post_attention_layernorm): Qwen2RMSNorm((64,), eps=1e-06)
          )
        )
        (norm): Qwen2RMSNorm((64,), eps=1e-06)
        (rotary_emb): Qwen2RotaryEmbedding()
      )
      (score): ModulesToSaveWrapper(
        (original_module): Linear(in_features=64, out_features=2, bias=False)
        (modules_to_save): ModuleDict(
          (default): Linear(in_features=64, out_features=2, bias=False)
        )
      )
    )
  )
)
```

# tiny-stable-diffusion-lora

## unet
```python
UNet2DConditionModel(
  (conv_in): Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (time_proj): Timesteps()
  (time_embedding): TimestepEmbedding(
    (linear_1): Linear(in_features=32, out_features=128, bias=True)
    (act): SiLU()
    (linear_2): Linear(in_features=128, out_features=128, bias=True)
  )
  (down_blocks): ModuleList(
    (0): DownBlock2D(
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 32, eps=1e-05, affine=True)
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=32, bias=True)
          (norm2): GroupNorm(32, 32, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (1): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 64, eps=1e-06, affine=True)
          (proj_in): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=64, out_features=64, bias=False)
                (to_k): lora.Linear(
                  (base_layer): Linear(in_features=64, out_features=64, bias=False)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=64, out_features=8, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=8, out_features=64, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (to_v): Linear(in_features=64, out_features=64, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=64, out_features=64, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=64, out_features=64, bias=False)
                (to_k): lora.Linear(
                  (base_layer): Linear(in_features=32, out_features=64, bias=False)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=32, out_features=8, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=8, out_features=64, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (to_v): Linear(in_features=32, out_features=64, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=64, out_features=64, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=64, out_features=512, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=256, out_features=64, bias=True)
                )
              )
            )
          )
          (proj_out): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 32, eps=1e-05, affine=True)
          (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
    )
  )
  (up_blocks): ModuleList(
    (0): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 64, eps=1e-06, affine=True)
          (proj_in): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=64, out_features=64, bias=False)
                (to_k): lora.Linear(
                  (base_layer): Linear(in_features=64, out_features=64, bias=False)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=64, out_features=8, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=8, out_features=64, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (to_v): Linear(in_features=64, out_features=64, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=64, out_features=64, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=64, out_features=64, bias=False)
                (to_k): lora.Linear(
                  (base_layer): Linear(in_features=32, out_features=64, bias=False)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=32, out_features=8, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=8, out_features=64, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (to_v): Linear(in_features=32, out_features=64, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=64, out_features=64, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=64, out_features=512, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=256, out_features=64, bias=True)
                )
              )
            )
          )
          (proj_out): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 96, eps=1e-05, affine=True)
          (conv1): Conv2d(96, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (1): UpBlock2D(
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 96, eps=1e-05, affine=True)
          (conv1): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=32, bias=True)
          (norm2): GroupNorm(32, 32, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1))
        )
        (1-2): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
          (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=128, out_features=32, bias=True)
          (norm2): GroupNorm(32, 32, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
  (mid_block): UNetMidBlock2DCrossAttn(
    (attentions): ModuleList(
      (0): Transformer2DModel(
        (norm): GroupNorm(32, 64, eps=1e-06, affine=True)
        (proj_in): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=64, out_features=64, bias=False)
              (to_k): lora.Linear(
                (base_layer): Linear(in_features=64, out_features=64, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=64, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=64, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (to_v): Linear(in_features=64, out_features=64, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=64, out_features=64, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=64, out_features=64, bias=False)
              (to_k): lora.Linear(
                (base_layer): Linear(in_features=32, out_features=64, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=32, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=64, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (to_v): Linear(in_features=32, out_features=64, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=64, out_features=64, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=64, out_features=512, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=256, out_features=64, bias=True)
              )
            )
          )
        )
        (proj_out): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (resnets): ModuleList(
      (0-1): 2 x ResnetBlock2D(
        (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
        (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
      )
    )
  )
  (conv_norm_out): GroupNorm(32, 32, eps=1e-05, affine=True)
  (conv_act): SiLU()
  (conv_out): Conv2d(32, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
```

## clip text model
```python
CLIPTextModel(
  (text_model): CLIPTextTransformer(
    (embeddings): CLIPTextEmbeddings(
      (token_embedding): Embedding(1000, 32)
      (position_embedding): Embedding(77, 32)
    )
    (encoder): CLIPEncoder(
      (layers): ModuleList(
        (0-4): 5 x CLIPEncoderLayer(
          (self_attn): CLIPAttention(
            (k_proj): lora.Linear(
              (base_layer): Linear(in_features=32, out_features=32, bias=True)
              (lora_dropout): ModuleDict(
                (default): Identity()
              )
              (lora_A): ModuleDict(
                (default): Linear(in_features=32, out_features=8, bias=False)
              )
              (lora_B): ModuleDict(
                (default): Linear(in_features=8, out_features=32, bias=False)
              )
              (lora_embedding_A): ParameterDict()
              (lora_embedding_B): ParameterDict()
              (lora_magnitude_vector): ModuleDict()
            )
            (v_proj): lora.Linear(
              (base_layer): Linear(in_features=32, out_features=32, bias=True)
              (lora_dropout): ModuleDict(
                (default): Identity()
              )
              (lora_A): ModuleDict(
                (default): Linear(in_features=32, out_features=8, bias=False)
              )
              (lora_B): ModuleDict(
                (default): Linear(in_features=8, out_features=32, bias=False)
              )
              (lora_embedding_A): ParameterDict()
              (lora_embedding_B): ParameterDict()
              (lora_magnitude_vector): ModuleDict()
            )
            (q_proj): lora.Linear(
              (base_layer): Linear(in_features=32, out_features=32, bias=True)
              (lora_dropout): ModuleDict(
                (default): Identity()
              )
              (lora_A): ModuleDict(
                (default): Linear(in_features=32, out_features=8, bias=False)
              )
              (lora_B): ModuleDict(
                (default): Linear(in_features=8, out_features=32, bias=False)
              )
              (lora_embedding_A): ParameterDict()
              (lora_embedding_B): ParameterDict()
              (lora_magnitude_vector): ModuleDict()
            )
            (out_proj): lora.Linear(
              (base_layer): Linear(in_features=32, out_features=32, bias=True)
              (lora_dropout): ModuleDict(
                (default): Identity()
              )
              (lora_A): ModuleDict(
                (default): Linear(in_features=32, out_features=8, bias=False)
              )
              (lora_B): ModuleDict(
                (default): Linear(in_features=8, out_features=32, bias=False)
              )
              (lora_embedding_A): ParameterDict()
              (lora_embedding_B): ParameterDict()
              (lora_magnitude_vector): ModuleDict()
            )
          )
          (layer_norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
          (mlp): CLIPMLP(
            (activation_fn): QuickGELUActivation()
            (fc1): Linear(in_features=32, out_features=37, bias=True)
            (fc2): Linear(in_features=37, out_features=32, bias=True)
          )
          (layer_norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (final_layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
  )
)
```

# 集成

加载这个新适配器并将其命名为 `adapter_name` ，然后使用 `set_adapters` 方法将其设置为当前活动的适配器，最后，可以调用 `disable_lora` 方法来恢复基础模型

添加适配器配置来指定如何适配模型参数。调用 `add_adapter()` 方法将该配置添加到基础模型

使用新训练的模型进行推理， `AutoModel` 类在后端使用 PEFT 将适配器权重和配置文件加载到基础预训练模型中

使用多个适配器，可以调用 `add_adapter()` 方法将适配器配置添加到基础模型。唯一的要求是适配器类型必须相同（不能混用 LoRA 和 LoHa 适配器）。

再次调用 `add_adapter()` 将新的适配器附加到基础模型。

使用 `set_adapter()` 来设置当前活动的适配器

`enable_adapters` 可用于再次启用适配器

# 在训练模型的 bug

之前在训练文本模型时经常 OOM ，问题出在数据集处理的 SEQ LEN 上，在 Qwen2 这样的模型里：

`create_causal_mask` 默认会生成一个 `(batch_size, num_heads, seq_len, seq_len)` 的 全量因果 `mask。`

如果 `seq_len` 很大（比如 4k、8k、16k），这个张量是 平方级别增长的。

如果只指定了 `padding='max_length'` ，但没有指定 `max_length` ，默认填充至模型接受的最大长度，自然会 OOM
å