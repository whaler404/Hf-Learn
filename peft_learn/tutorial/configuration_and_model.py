# lora config
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)

print(lora_config)
# {
#   "task_type": "CAUSAL_LM",
#   "peft_type": "LORA",
#   "auto_mapping": null,
#   "base_model_name_or_path": null,
#   "revision": null,
#   "inference_mode": false,
#   "r": 16,
#   "target_modules": [
#     "o_proj",
#     "v_proj",
#     "q_proj",
#     "k_proj"
#   ],
#   "exclude_modules": null,
#   "lora_alpha": 32,
#   "lora_dropout": 0.05,
#   "fan_in_fan_out": false,
#   "bias": "none",
#   "use_rslora": false,
#   "modules_to_save": null,
#   "init_lora_weights": true,
#   "layers_to_transform": null,
#   "layers_pattern": null,
#   "rank_pattern": {},
#   "alpha_pattern": {},
#   "megatron_config": null,
#   "megatron_core": "megatron.core",
#   "trainable_token_indices": null,
#   "loftq_config": {},
#   "eva_config": null,
#   "corda_config": null,
#   "use_dora": false,
#   "use_qalora": false,
#   "qalora_group_size": 16,
#   "layer_replication": null,
#   "runtime_config": {
#     "ephemeral_gpu_offload": false
#   },
#   "lora_bias": false,
#   "target_parameters": null
# }

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("wheeler404/qwen2-tiny")

from peft import get_peft_model

lora_model = get_peft_model(model, lora_config)

lora_model.print_trainable_parameters()
# trainable params: 26,624 || all params: 13,528,016 || trainable%: 0.1968

print(lora_model)
# PeftModelForCausalLM(
#   (base_model): LoraModel(
#     (model): Qwen2ForCausalLM(
#       (model): Qwen2Model(
#         (embed_tokens): Embedding(151936, 64)
#         (layers): ModuleList(
#           (0-3): 4 x Qwen2DecoderLayer(
#             (self_attn): Qwen2Attention(
#               (q_proj): lora.Linear(
#                 (base_layer): Linear(in_features=64, out_features=60, bias=True)
#                 (lora_dropout): ModuleDict(
#                   (default): Dropout(p=0.05, inplace=False)
#                 )
#                 (lora_A): ModuleDict(
#                   (default): Linear(in_features=64, out_features=16, bias=False)
#                 )
#                 (lora_B): ModuleDict(
#                   (default): Linear(in_features=16, out_features=60, bias=False)
#                 )
#                 (lora_embedding_A): ParameterDict()
#                 (lora_embedding_B): ParameterDict()
#                 (lora_magnitude_vector): ModuleDict()
#               )
#               (k_proj): lora.Linear(
#                 (base_layer): Linear(in_features=64, out_features=20, bias=True)
#                 (lora_dropout): ModuleDict(
#                   (default): Dropout(p=0.05, inplace=False)
#                 )
#                 (lora_A): ModuleDict(
#                   (default): Linear(in_features=64, out_features=16, bias=False)
#                 )
#                 (lora_B): ModuleDict(
#                   (default): Linear(in_features=16, out_features=20, bias=False)
#                 )
#                 (lora_embedding_A): ParameterDict()
#                 (lora_embedding_B): ParameterDict()
#                 (lora_magnitude_vector): ModuleDict()
#               )
#               (v_proj): lora.Linear(
#                 (base_layer): Linear(in_features=64, out_features=20, bias=True)
#                 (lora_dropout): ModuleDict(
#                   (default): Dropout(p=0.05, inplace=False)
#                 )
#                 (lora_A): ModuleDict(
#                   (default): Linear(in_features=64, out_features=16, bias=False)
#                 )
#                 (lora_B): ModuleDict(
#                   (default): Linear(in_features=16, out_features=20, bias=False)
#                 )
#                 (lora_embedding_A): ParameterDict()
#                 (lora_embedding_B): ParameterDict()
#                 (lora_magnitude_vector): ModuleDict()
#               )
#               (o_proj): lora.Linear(
#                 (base_layer): Linear(in_features=60, out_features=64, bias=False)
#                 (lora_dropout): ModuleDict(
#                   (default): Dropout(p=0.05, inplace=False)
#                 )
#                 (lora_A): ModuleDict(
#                   (default): Linear(in_features=60, out_features=16, bias=False)
#                 )
#                 (lora_B): ModuleDict(
#                   (default): Linear(in_features=16, out_features=64, bias=False)
#                 )
#                 (lora_embedding_A): ParameterDict()
#                 (lora_embedding_B): ParameterDict()
#                 (lora_magnitude_vector): ModuleDict()
#               )
#             )
#             (mlp): Qwen2MLP(
#               (gate_proj): Linear(in_features=64, out_features=4864, bias=False)
#               (up_proj): Linear(in_features=64, out_features=4864, bias=False)
#               (down_proj): Linear(in_features=4864, out_features=64, bias=False)
#               (act_fn): SiLU()
#             )
#             (input_layernorm): Qwen2RMSNorm((64,), eps=1e-06)
#             (post_attention_layernorm): Qwen2RMSNorm((64,), eps=1e-06)
#           )
#         )
#         (norm): Qwen2RMSNorm((64,), eps=1e-06)
#         (rotary_emb): Qwen2RotaryEmbedding()
#       )
#       (lm_head): Linear(in_features=64, out_features=151936, bias=False)
#     )
#   )
# )

from torchinfo import summary
print(summary(lora_model))
# ==========================================================================================
# Layer (type:depth-idx)                                            Param #
# ==========================================================================================
# PeftModelForCausalLM                                              --
# ├─LoraModel: 1-1                                                  --
# │    └─Qwen2ForCausalLM: 2-1                                      --
# │    │    └─Qwen2Model: 3-1                                       13,528,016
# │    │    └─Linear: 3-2                                           (9,723,904)
# ==========================================================================================
# Total params: 23,251,920
# Trainable params: 26,624
# Non-trainable params: 23,225,296
# ==========================================================================================

# p-tuning config

from peft import PromptEncoderConfig, TaskType

p_tuning_config = PromptEncoderConfig(
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=128,
    # num_attention_heads=16,
    # num_layers=24,
    # num_transformer_submodules=1,
    num_virtual_tokens=20,
    token_dim=1024,
    task_type=TaskType.CAUSAL_LM
)

print(p_tuning_config)
# {
#   "task_type": "CAUSAL_LM",
#   "peft_type": "P_TUNING",
#   "auto_mapping": null,
#   "base_model_name_or_path": null,
#   "revision": null,
#   "inference_mode": false,
#   "num_virtual_tokens": 20,
#   "token_dim": 1024,
#   "num_transformer_submodules": null,
#   "num_attention_heads": null,
#   "num_layers": null,
#   "modules_to_save": null,
#   "encoder_reparameterization_type": "MLP",
#   "encoder_hidden_size": 128,
#   "encoder_num_layers": 2,
#   "encoder_dropout": 0.0
# }

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("wheeler404/qwen2-tiny")

from peft import get_peft_model

p_tuning_model = get_peft_model(model, p_tuning_config)

p_tuning_model.print_trainable_parameters()
# trainable params: 300,288 || all params: 13,801,680 || trainable%: 2.1757

print(p_tuning_model)
# PeftModelForSequenceClassification(
#   (base_model): Qwen2ForCausalLM(
#     (model): Qwen2Model(
#       (embed_tokens): Embedding(151936, 64)
#       (layers): ModuleList(
#         (0-3): 4 x Qwen2DecoderLayer(
#           (self_attn): Qwen2Attention(
#             (q_proj): Linear(in_features=64, out_features=60, bias=True)
#             (k_proj): Linear(in_features=64, out_features=20, bias=True)
#             (v_proj): Linear(in_features=64, out_features=20, bias=True)
#             (o_proj): Linear(in_features=60, out_features=64, bias=False)
#           )
#           (mlp): Qwen2MLP(
#             (gate_proj): Linear(in_features=64, out_features=4864, bias=False)
#             (up_proj): Linear(in_features=64, out_features=4864, bias=False)
#             (down_proj): Linear(in_features=4864, out_features=64, bias=False)
#             (act_fn): SiLU()
#           )
#           (input_layernorm): Qwen2RMSNorm((64,), eps=1e-06)
#           (post_attention_layernorm): Qwen2RMSNorm((64,), eps=1e-06)
#         )
#       )
#       (norm): Qwen2RMSNorm((64,), eps=1e-06)
#       (rotary_emb): Qwen2RotaryEmbedding()
#     )
#     (lm_head): Linear(in_features=64, out_features=151936, bias=False)
#   )
#   (prompt_encoder): ModuleDict(
#     (default): PromptEncoder(
#       (embedding): Embedding(20, 1024)
#       (mlp_head): Sequential(
#         (0): Linear(in_features=1024, out_features=128, bias=True)
#         (1): ReLU()
#         (2): Linear(in_features=128, out_features=128, bias=True)
#         (3): ReLU()
#         (4): Linear(in_features=128, out_features=1024, bias=True)
#       )
#     )
#   )
#   (word_embeddings): Embedding(151936, 64)
# )

from torchinfo import summary
print(summary(p_tuning_model))
# ===========================================================================
# Layer (type:depth-idx)                             Param #
# ===========================================================================
# PeftModelForSequenceClassification                 --
# ├─Qwen2ForCausalLM: 1-1                            --
# │    └─Qwen2Model: 2-1                             --
# │    │    └─Embedding: 3-1                         (9,723,904)
# │    │    └─ModuleList: 3-2                        (3,777,424)
# │    │    └─Qwen2RMSNorm: 3-3                      (64)
# │    │    └─Qwen2RotaryEmbedding: 3-4              --
# │    └─Linear: 2-2                                 (9,723,904)
# ├─ModuleDict: 1-2                                  --
# │    └─PromptEncoder: 2-3                          --
# │    │    └─Embedding: 3-5                         20,480
# │    │    └─Sequential: 3-6                        279,808
# ├─Embedding: 1-3                                   (recursive)
# ===========================================================================
# Total params: 23,525,584
# Trainable params: 300,288
# Non-trainable params: 23,225,296
# ===========================================================================