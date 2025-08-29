# https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

# from diffusers import StableDiffusionPipeline

# pipeline = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None)

from diffusers.models.unets import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    "hf-internal-testing/tiny-stable-diffusion-torch",
    subfolder="unet",
    # torch_dtype=None,  # Use default dtype
)

from peft import TaskType, LoraConfig

from peft import get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_k"],
    lora_dropout=0.1,
    init_lora_weights="gaussian",
)

# 两种写法都可以
# unet=get_peft_model(unet,lora_config)
unet.add_adapter(lora_config)

print(unet)
# UNet2DConditionModel(
#   (conv_in): Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (time_proj): Timesteps()
#   (time_embedding): TimestepEmbedding(
#     (linear_1): Linear(in_features=32, out_features=128, bias=True)
#     (act): SiLU()
#     (linear_2): Linear(in_features=128, out_features=128, bias=True)
#   )
#   (down_blocks): ModuleList(
#     (0): DownBlock2D(
#       (resnets): ModuleList(
#         (0-1): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 32, eps=1e-05, affine=True)
#           (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=128, out_features=32, bias=True)
#           (norm2): GroupNorm(32, 32, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample2D(
#           (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (1): CrossAttnDownBlock2D(
#       (attentions): ModuleList(
#         (0-1): 2 x Transformer2DModel(
#           (norm): GroupNorm(32, 64, eps=1e-06, affine=True)
#           (proj_in): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=64, out_features=64, bias=False)
#                 (to_k): lora.Linear(
#                   (base_layer): Linear(in_features=64, out_features=64, bias=False)
#                   (lora_dropout): ModuleDict(
#                     (default): Dropout(p=0.1, inplace=False)
#                   )
#                   (lora_A): ModuleDict(
#                     (default): Linear(in_features=64, out_features=8, bias=False)
#                   )
#                   (lora_B): ModuleDict(
#                     (default): Linear(in_features=8, out_features=64, bias=False)
#                   )
#                   (lora_embedding_A): ParameterDict()
#                   (lora_embedding_B): ParameterDict()
#                   (lora_magnitude_vector): ModuleDict()
#                 )
#                 (to_v): Linear(in_features=64, out_features=64, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=64, out_features=64, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=64, out_features=64, bias=False)
#                 (to_k): lora.Linear(
#                   (base_layer): Linear(in_features=32, out_features=64, bias=False)
#                   (lora_dropout): ModuleDict(
#                     (default): Dropout(p=0.1, inplace=False)
#                   )
#                   (lora_A): ModuleDict(
#                     (default): Linear(in_features=32, out_features=8, bias=False)
#                   )
#                   (lora_B): ModuleDict(
#                     (default): Linear(in_features=8, out_features=64, bias=False)
#                   )
#                   (lora_embedding_A): ParameterDict()
#                   (lora_embedding_B): ParameterDict()
#                   (lora_magnitude_vector): ModuleDict()
#                 )
#                 (to_v): Linear(in_features=32, out_features=64, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=64, out_features=64, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=64, out_features=512, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=256, out_features=64, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 32, eps=1e-05, affine=True)
#           (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
#           (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
#           (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
#           (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#     )
#   )
#   (up_blocks): ModuleList(
#     (0): CrossAttnUpBlock2D(
#       (attentions): ModuleList(
#         (0-2): 3 x Transformer2DModel(
#           (norm): GroupNorm(32, 64, eps=1e-06, affine=True)
#           (proj_in): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=64, out_features=64, bias=False)
#                 (to_k): lora.Linear(
#                   (base_layer): Linear(in_features=64, out_features=64, bias=False)
#                   (lora_dropout): ModuleDict(
#                     (default): Dropout(p=0.1, inplace=False)
#                   )
#                   (lora_A): ModuleDict(
#                     (default): Linear(in_features=64, out_features=8, bias=False)
#                   )
#                   (lora_B): ModuleDict(
#                     (default): Linear(in_features=8, out_features=64, bias=False)
#                   )
#                   (lora_embedding_A): ParameterDict()
#                   (lora_embedding_B): ParameterDict()
#                   (lora_magnitude_vector): ModuleDict()
#                 )
#                 (to_v): Linear(in_features=64, out_features=64, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=64, out_features=64, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=64, out_features=64, bias=False)
#                 (to_k): lora.Linear(
#                   (base_layer): Linear(in_features=32, out_features=64, bias=False)
#                   (lora_dropout): ModuleDict(
#                     (default): Dropout(p=0.1, inplace=False)
#                   )
#                   (lora_A): ModuleDict(
#                     (default): Linear(in_features=32, out_features=8, bias=False)
#                   )
#                   (lora_B): ModuleDict(
#                     (default): Linear(in_features=8, out_features=64, bias=False)
#                   )
#                   (lora_embedding_A): ParameterDict()
#                   (lora_embedding_B): ParameterDict()
#                   (lora_magnitude_vector): ModuleDict()
#                 )
#                 (to_v): Linear(in_features=32, out_features=64, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=64, out_features=64, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=64, out_features=512, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=256, out_features=64, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0-1): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 128, eps=1e-05, affine=True)
#           (conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
#           (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): ResnetBlock2D(
#           (norm1): GroupNorm(32, 96, eps=1e-05, affine=True)
#           (conv1): Conv2d(96, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
#           (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample2D(
#           (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (1): UpBlock2D(
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 96, eps=1e-05, affine=True)
#           (conv1): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=128, out_features=32, bias=True)
#           (norm2): GroupNorm(32, 32, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1-2): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
#           (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=128, out_features=32, bias=True)
#           (norm2): GroupNorm(32, 32, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#     )
#   )
#   (mid_block): UNetMidBlock2DCrossAttn(
#     (attentions): ModuleList(
#       (0): Transformer2DModel(
#         (norm): GroupNorm(32, 64, eps=1e-06, affine=True)
#         (proj_in): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
#         (transformer_blocks): ModuleList(
#           (0): BasicTransformerBlock(
#             (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#             (attn1): Attention(
#               (to_q): Linear(in_features=64, out_features=64, bias=False)
#               (to_k): lora.Linear(
#                 (base_layer): Linear(in_features=64, out_features=64, bias=False)
#                 (lora_dropout): ModuleDict(
#                   (default): Dropout(p=0.1, inplace=False)
#                 )
#                 (lora_A): ModuleDict(
#                   (default): Linear(in_features=64, out_features=8, bias=False)
#                 )
#                 (lora_B): ModuleDict(
#                   (default): Linear(in_features=8, out_features=64, bias=False)
#                 )
#                 (lora_embedding_A): ParameterDict()
#                 (lora_embedding_B): ParameterDict()
#                 (lora_magnitude_vector): ModuleDict()
#               )
#               (to_v): Linear(in_features=64, out_features=64, bias=False)
#               (to_out): ModuleList(
#                 (0): Linear(in_features=64, out_features=64, bias=True)
#                 (1): Dropout(p=0.0, inplace=False)
#               )
#             )
#             (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#             (attn2): Attention(
#               (to_q): Linear(in_features=64, out_features=64, bias=False)
#               (to_k): lora.Linear(
#                 (base_layer): Linear(in_features=32, out_features=64, bias=False)
#                 (lora_dropout): ModuleDict(
#                   (default): Dropout(p=0.1, inplace=False)
#                 )
#                 (lora_A): ModuleDict(
#                   (default): Linear(in_features=32, out_features=8, bias=False)
#                 )
#                 (lora_B): ModuleDict(
#                   (default): Linear(in_features=8, out_features=64, bias=False)
#                 )
#                 (lora_embedding_A): ParameterDict()
#                 (lora_embedding_B): ParameterDict()
#                 (lora_magnitude_vector): ModuleDict()
#               )
#               (to_v): Linear(in_features=32, out_features=64, bias=False)
#               (to_out): ModuleList(
#                 (0): Linear(in_features=64, out_features=64, bias=True)
#                 (1): Dropout(p=0.0, inplace=False)
#               )
#             )
#             (norm3): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#             (ff): FeedForward(
#               (net): ModuleList(
#                 (0): GEGLU(
#                   (proj): Linear(in_features=64, out_features=512, bias=True)
#                 )
#                 (1): Dropout(p=0.0, inplace=False)
#                 (2): Linear(in_features=256, out_features=64, bias=True)
#               )
#             )
#           )
#         )
#         (proj_out): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
#       )
#     )
#     (resnets): ModuleList(
#       (0-1): 2 x ResnetBlock2D(
#         (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
#         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (time_emb_proj): Linear(in_features=128, out_features=64, bias=True)
#         (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
#         (dropout): Dropout(p=0.0, inplace=False)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (nonlinearity): SiLU()
#       )
#     )
#   )
#   (conv_norm_out): GroupNorm(32, 32, eps=1e-05, affine=True)
#   (conv_act): SiLU()
#   (conv_out): Conv2d(32, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# )

from transformers.models.clip import CLIPTextModel

text_encoder = CLIPTextModel.from_pretrained(
    "hf-internal-testing/tiny-stable-diffusion-torch",
    subfolder="text_encoder",
    # torch_dtype=None,  # Use default dtype
)

text_lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    init_lora_weights="gaussian",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
)

text_encoder.add_adapter(text_lora_config)

print(text_encoder)
# CLIPTextModel(
#   (text_model): CLIPTextTransformer(
#     (embeddings): CLIPTextEmbeddings(
#       (token_embedding): Embedding(1000, 32)
#       (position_embedding): Embedding(77, 32)
#     )
#     (encoder): CLIPEncoder(
#       (layers): ModuleList(
#         (0-4): 5 x CLIPEncoderLayer(
#           (self_attn): CLIPAttention(
#             (k_proj): lora.Linear(
#               (base_layer): Linear(in_features=32, out_features=32, bias=True)
#               (lora_dropout): ModuleDict(
#                 (default): Identity()
#               )
#               (lora_A): ModuleDict(
#                 (default): Linear(in_features=32, out_features=8, bias=False)
#               )
#               (lora_B): ModuleDict(
#                 (default): Linear(in_features=8, out_features=32, bias=False)
#               )
#               (lora_embedding_A): ParameterDict()
#               (lora_embedding_B): ParameterDict()
#               (lora_magnitude_vector): ModuleDict()
#             )
#             (v_proj): lora.Linear(
#               (base_layer): Linear(in_features=32, out_features=32, bias=True)
#               (lora_dropout): ModuleDict(
#                 (default): Identity()
#               )
#               (lora_A): ModuleDict(
#                 (default): Linear(in_features=32, out_features=8, bias=False)
#               )
#               (lora_B): ModuleDict(
#                 (default): Linear(in_features=8, out_features=32, bias=False)
#               )
#               (lora_embedding_A): ParameterDict()
#               (lora_embedding_B): ParameterDict()
#               (lora_magnitude_vector): ModuleDict()
#             )
#             (q_proj): lora.Linear(
#               (base_layer): Linear(in_features=32, out_features=32, bias=True)
#               (lora_dropout): ModuleDict(
#                 (default): Identity()
#               )
#               (lora_A): ModuleDict(
#                 (default): Linear(in_features=32, out_features=8, bias=False)
#               )
#               (lora_B): ModuleDict(
#                 (default): Linear(in_features=8, out_features=32, bias=False)
#               )
#               (lora_embedding_A): ParameterDict()
#               (lora_embedding_B): ParameterDict()
#               (lora_magnitude_vector): ModuleDict()
#             )
#             (out_proj): lora.Linear(
#               (base_layer): Linear(in_features=32, out_features=32, bias=True)
#               (lora_dropout): ModuleDict(
#                 (default): Identity()
#               )
#               (lora_A): ModuleDict(
#                 (default): Linear(in_features=32, out_features=8, bias=False)
#               )
#               (lora_B): ModuleDict(
#                 (default): Linear(in_features=8, out_features=32, bias=False)
#               )
#               (lora_embedding_A): ParameterDict()
#               (lora_embedding_B): ParameterDict()
#               (lora_magnitude_vector): ModuleDict()
#             )
#           )
#           (layer_norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
#           (mlp): CLIPMLP(
#             (activation_fn): QuickGELUActivation()
#             (fc1): Linear(in_features=32, out_features=37, bias=True)
#             (fc2): Linear(in_features=37, out_features=32, bias=True)
#           )
#           (layer_norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#     )
#     (final_layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
#   )
# )

# 下面是 diffusers/tiny-stable-diffusion-torch 的配置信息

# model_index.json 
# {
#   "_class_name": "StableDiffusionPipeline",
#   "_diffusers_version": "0.7.0.dev0",
#   "feature_extractor": [
#     "transformers",
#     "CLIPImageProcessor"
#   ],
#   "requires_safety_checker": false,
#   "safety_checker": [
#     "stable_diffusion",
#     "StableDiffusionSafetyChecker"
#   ],
#   "scheduler": [
#     "diffusers",
#     "PNDMScheduler"
#   ],
#   "text_encoder": [
#     "transformers",
#     "CLIPTextModel"
#   ],
#   "tokenizer": [
#     "transformers",
#     "CLIPTokenizer"
#   ],
#   "unet": [
#     "diffusers",
#     "UNet2DConditionModel"
#   ],
#   "vae": [
#     "diffusers",
#     "AutoencoderKL"
#   ]
# }

# text_encoder/config.json 
# {
#   "architectures": [
#     "CLIPTextModel"
#   ],
#   "attention_dropout": 0.0,
#   "bos_token_id": 0,
#   "dropout": 0.0,
#   "eos_token_id": 2,
#   "hidden_act": "quick_gelu",
#   "hidden_size": 32,
#   "initializer_factor": 1.0,
#   "initializer_range": 0.02,
#   "intermediate_size": 37,
#   "layer_norm_eps": 1e-05,
#   "max_position_embeddings": 77,
#   "model_type": "clip_text_model",
#   "num_attention_heads": 4,
#   "num_hidden_layers": 5,
#   "pad_token_id": 1,
#   "torch_dtype": "float32",
#   "transformers_version": "4.22.2",
#   "vocab_size": 1000
# }

# preprocessor_config.json 
# {
#   "crop_size": 224,
#   "do_center_crop": true,
#   "do_convert_rgb": true,
#   "do_normalize": true,
#   "do_resize": true,
#   "feature_extractor_type": "CLIPFeatureExtractor",
#   "image_mean": [
#     0.48145466,
#     0.4578275,
#     0.40821073
#   ],
#   "image_std": [
#     0.26862954,
#     0.26130258,
#     0.27577711
#   ],
#   "resample": 3,
#   "size": 224
# }

# scheduler_config.json 
# {
#   "_class_name": "PNDMScheduler",
#   "_diffusers_version": "0.7.0.dev0",
#   "beta_end": 0.012,
#   "beta_schedule": "scaled_linear",
#   "beta_start": 0.00085,
#   "num_train_timesteps": 1000,
#   "set_alpha_to_one": false,
#   "skip_prk_steps": true,
#   "steps_offset": 1,
#   "trained_betas": null,
#   "clip_sample": false
# }

# unet/config.json 
# {
#   "_class_name": "UNet2DConditionModel",
#   "_diffusers_version": "0.7.0.dev0",
#   "act_fn": "silu",
#   "attention_head_dim": 8,
#   "block_out_channels": [
#     32,
#     64
#   ],
#   "center_input_sample": false,
#   "cross_attention_dim": 32,
#   "down_block_types": [
#     "DownBlock2D",
#     "CrossAttnDownBlock2D"
#   ],
#   "downsample_padding": 1,
#   "flip_sin_to_cos": true,
#   "freq_shift": 0,
#   "in_channels": 4,
#   "layers_per_block": 2,
#   "mid_block_scale_factor": 1,
#   "norm_eps": 1e-05,
#   "norm_num_groups": 32,
#   "out_channels": 4,
#   "sample_size": 32,
#   "up_block_types": [
#     "CrossAttnUpBlock2D",
#     "UpBlock2D"
#   ]
# }

# vae/config.json 
# {
#   "_class_name": "AutoencoderKL",
#   "_diffusers_version": "0.7.0.dev0",
#   "act_fn": "silu",
#   "block_out_channels": [
#     32,
#     64
#   ],
#   "down_block_types": [
#     "DownEncoderBlock2D",
#     "DownEncoderBlock2D"
#   ],
#   "in_channels": 3,
#   "latent_channels": 4,
#   "layers_per_block": 1,
#   "norm_num_groups": 32,
#   "out_channels": 3,
#   "sample_size": 128,
#   "up_block_types": [
#     "UpDecoderBlock2D",
#     "UpDecoderBlock2D"
#   ]
# }