from timm.models.tiny_vit import TinyVit

model = TinyVit(
    in_chans=3,
    num_classes=10,
    embed_dims=(96, 192,),
    depths=(1, 1,),
    num_heads=(3, 6,),
    window_sizes=(7, 7,),
)

print([(n, type(m)) for n, m in model.named_modules() if "mlp.fc" in n or "head.fc" in n])

from peft import LoraConfig, get_peft_model
config = LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=["head.fc"])
# [
#   [
#     "stages.1.blocks.0.mlp.fc1",
#     "torch.nn.modules.linear.Linear"
#   ],
#   [
#     "stages.1.blocks.0.mlp.fc2",
#     "torch.nn.modules.linear.Linear"
#   ],
#   [
#     "head.fc",
#     "torch.nn.modules.linear.Linear"
#   ]
# ]

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# trainable params: 17,290 || all params: 647,018 || trainable%: 2.6723

print([(n, type(m)) for n, m in peft_model.named_modules() if "mlp.fc" in n or "head.fc" in n])
# [
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1",
#     "peft.tuners.lora.layer.Linear"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.base_layer",
#     "torch.nn.modules.linear.Linear"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.lora_dropout",
#     "torch.nn.modules.container.ModuleDict"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.lora_dropout.default",
#     "torch.nn.modules.linear.Identity"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.lora_A",
#     "torch.nn.modules.container.ModuleDict"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.lora_A.default",
#     "torch.nn.modules.linear.Linear"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.lora_B",
#     "torch.nn.modules.container.ModuleDict"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.lora_B.default",
#     "torch.nn.modules.linear.Linear"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.lora_embedding_A",
#     "torch.nn.modules.container.ParameterDict"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.lora_embedding_B",
#     "torch.nn.modules.container.ParameterDict"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.lora_magnitude_vector",
#     "torch.nn.modules.container.ModuleDict"
#   ],
#   ...
#   [
#     "base_model.model.head.fc",
#     "peft.utils.other.ModulesToSaveWrapper"
#   ],
#   [
#     "base_model.model.head.fc.original_module",
#     "torch.nn.modules.linear.Linear"
#   ],
#   [
#     "base_model.model.head.fc.modules_to_save",
#     "torch.nn.modules.container.ModuleDict"
#   ],
#   [
#     "base_model.model.head.fc.modules_to_save.default",
#     "torch.nn.modules.linear.Linear"
#   ]
# ]

print([(n, type(m)) for n, m in peft_model.named_parameters() if "mlp.fc" in n or "head.fc" in n])
# [
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.base_layer.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.base_layer.bias",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.lora_A.default.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc1.lora_B.default.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc2.base_layer.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc2.base_layer.bias",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc2.lora_A.default.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.stages.1.blocks.0.mlp.fc2.lora_B.default.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.head.fc.original_module.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.head.fc.original_module.bias",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.head.fc.modules_to_save.default.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.head.fc.modules_to_save.default.bias",
#     "torch.nn.parameter.Parameter"
#   ]
# ]