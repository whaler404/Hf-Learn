import torch
from transformers import ViTForImageClassification

# 1️⃣ 先加载 LightningModule
from prepare_teacher_model import LitViT  # 你的 Lightning 模型类定义所在文件
ckpt_path = "trainer_output/distill_vit/teacher-vit-36-0.9348.ckpt"
lit_model = LitViT.load_from_checkpoint(ckpt_path)

# 2️⃣ 取出内部的 transformers 模型
vit_model = lit_model.model

# 3️⃣ 现在 vit_model 就是标准的 Hugging Face ViTForImageClassification 实例
# 可以直接使用 Transformers API
vit_model.save_pretrained("trainer_output/distill_vit/teacher-vit")  # 保存为 HF 格式
