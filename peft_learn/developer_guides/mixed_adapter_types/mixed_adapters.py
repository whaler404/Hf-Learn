from peft import PeftMixedModel, LoraConfig

adapter_1 = LoraConfig()

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("wheeler404/qwen2-tiny")

peft_model = PeftMixedModel.from_pretrained(
    model,
    "wheeler404/qwen2-tiny-lora",
    adapter_name="adapter_1",
)

adapter_2 = LoraConfig()

peft_model.add_adapter(peft_config=adapter_2, adapter_name="adapter_2")
peft_model.set_adapter(["adapter_1", "adapter_2"])

print(peft_model)
# PeftMixedModel(
#   (base_model): MixedModel(
#     (model): Qwen2ForCausalLM(
#       (model): Qwen2Model(
#         (embed_tokens): Embedding(151936, 64)
#         (layers): ModuleList(
#           (0-3): 4 x Qwen2DecoderLayer(
#             (self_attn): Qwen2Attention(
#               (q_proj): lora.Linear(
#                 (base_layer): Linear(in_features=64, out_features=60, bias=True)
#                 (lora_dropout): ModuleDict(
#                   (adapter_1): Dropout(p=0.1, inplace=False)
#                   (adapter_2): Identity()
#                 )
#                 (lora_A): ModuleDict(
#                   (adapter_1): Linear(in_features=64, out_features=8, bias=False)
#                   (adapter_2): Linear(in_features=64, out_features=8, bias=False)
#                 )
#                 (lora_B): ModuleDict(
#                   (adapter_1): Linear(in_features=8, out_features=60, bias=False)
#                   (adapter_2): Linear(in_features=8, out_features=60, bias=False)
#                 )
#                 (lora_embedding_A): ParameterDict()
#                 (lora_embedding_B): ParameterDict()
#                 (lora_magnitude_vector): ModuleDict()
#               )
#               (k_proj): Linear(in_features=64, out_features=20, bias=True)
#               (v_proj): lora.Linear(
#                 (base_layer): Linear(in_features=64, out_features=20, bias=True)
#                 (lora_dropout): ModuleDict(
#                   (adapter_1): Dropout(p=0.1, inplace=False)
#                   (adapter_2): Identity()
#                 )
#                 (lora_A): ModuleDict(
#                   (adapter_1): Linear(in_features=64, out_features=8, bias=False)
#                   (adapter_2): Linear(in_features=64, out_features=8, bias=False)
#                 )
#                 (lora_B): ModuleDict(
#                   (adapter_1): Linear(in_features=8, out_features=20, bias=False)
#                   (adapter_2): Linear(in_features=8, out_features=20, bias=False)
#                 )
#                 (lora_embedding_A): ParameterDict()
#                 (lora_embedding_B): ParameterDict()
#                 (lora_magnitude_vector): ModuleDict()
#               )
#               (o_proj): Linear(in_features=60, out_features=64, bias=False)
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