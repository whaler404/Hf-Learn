from torch import nn


class MLP(nn.Module):
    def __init__(self, num_units_hidden=64):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(20, num_units_hidden),
            nn.ReLU(),
            nn.Linear(num_units_hidden, num_units_hidden),
            nn.ReLU(),
            nn.Linear(num_units_hidden, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, X):
        return self.seq(X)

print([(n,type(m))for n,m in MLP().named_modules()])
# [
#   [
#     "",
#     "__main__.MLP"
#   ],
#   [
#     "seq",
#     "torch.nn.modules.container.Sequential"
#   ],
#   [
#     "seq.0",
#     "torch.nn.modules.linear.Linear"
#   ],
#   [
#     "seq.1",
#     "torch.nn.modules.activation.ReLU"
#   ],
#   [
#     "seq.2",
#     "torch.nn.modules.linear.Linear"
#   ],
#   [
#     "seq.3",
#     "torch.nn.modules.activation.ReLU"
#   ],
#   [
#     "seq.4",
#     "torch.nn.modules.linear.Linear"
#   ],
#   [
#     "seq.5",
#     "torch.nn.modules.activation.LogSoftmax"
#   ]
# ]

print([(n,type(m))for n,m in MLP().named_parameters()])
# [
#   [
#     "seq.0.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "seq.0.bias",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "seq.2.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "seq.2.bias",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "seq.4.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "seq.4.bias",
#     "torch.nn.parameter.Parameter"
#   ]
# ]

from peft import LoraConfig

config = LoraConfig(
    target_modules=["seq.0", "seq.2"],
    modules_to_save=["seq.4"],
)

from peft import get_peft_model
peft_model = get_peft_model(MLP(), config)
peft_model.print_trainable_parameters()
# trainable params: 1,826 || all params: 7,460 || trainable%: 24.4772

print([(n,type(m))for n,m in peft_model.named_parameters()])
# [
#   [
#     "base_model.model.seq.0.base_layer.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.seq.0.base_layer.bias",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.seq.0.lora_A.default.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.seq.0.lora_B.default.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.seq.2.base_layer.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.seq.2.base_layer.bias",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.seq.2.lora_A.default.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.seq.2.lora_B.default.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.seq.4.original_module.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.seq.4.original_module.bias",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.seq.4.modules_to_save.default.weight",
#     "torch.nn.parameter.Parameter"
#   ],
#   [
#     "base_model.model.seq.4.modules_to_save.default.bias",
#     "torch.nn.parameter.Parameter"
#   ]
# ]