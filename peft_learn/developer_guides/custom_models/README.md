将 LoRA 应用于多层感知器、来自 timm 库的计算机视觉模型或新的 🤗 Transformers 架构

# mlp

用户自行选择层。要确定要调整的层的名称，将 LoRA 应用于输入层和隐藏层，即 'seq.0' 和 'seq.2' 。此外，假设在不使用 LoRA 的情况下更新输出层，即 'seq.4'

# timm

由于 LoRA 支持 2D 卷积层，并且这些层是该模型的主要构建块，因此应该将 LoRA 应用于 2D 卷积层。2D 卷积层有诸如 "stages.0.blocks.0.mlp.fc1" 之类的名称， "stages.0.blocks.0.mlp.fc2" ，编写一个正则表达式来匹配层名称，

# transformers

# 验证参数和层

使用 print_trainable_parameters() 方法检查可训练参数的比例。

使用 targeted_module_names 属性列出已适配的每个模块的名称

可以将 LoRA 应用于 nn.Linear 和 nn.Conv2d 层，但不能应用于 nn.LSTM 等层