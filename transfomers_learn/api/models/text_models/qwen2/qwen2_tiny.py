from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM

# config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1")
config = AutoConfig.from_pretrained("./transfomers_learn/api/models/text_models/qwen2/config.json")
model = AutoModelForCausalLM.from_config(config)

from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

from torchinfo import summary
summary(model)


# 测试模型
print("\n正在测试修改后的模型...")
test_input = "Hello, how are you?"
inputs = tokenizer(test_input, return_tensors="pt")

import torch
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成的文本: {generated_text}")

# 保存修改后的模型到本地
local_save_path = "./trainer_output/qwen2-tiny"
print(f"\n正在保存模型到本地: {local_save_path}")
model.save_pretrained(local_save_path)
tokenizer.save_pretrained(local_save_path)
print("本地保存完成！")

# 推送到Hugging Face Hub
def push_to_hub():
    """推送模型到Hugging Face Hub"""
    try:
        from huggingface_hub import login
        
        # 注意：您需要先登录 huggingface-cli login 或设置 HF_TOKEN 环境变量
        print("\n准备推送到Hugging Face Hub...")
        print("\n准备推送到Hugging Face Hub...")
        
        # 设置Hub仓库名称（请根据需要修改）
        hub_model_name = "qwen2-tiny"  # 修改为您想要的名称
        
        import os
        login(token=os.getenv("HF_TOKEN"))

        # 推送模型
        model.push_to_hub(
            repo_id=hub_model_name,
            commit_message="Add Qwen2.5-tiny model with single hidden layer",
            private=False  # 设置为True如果您想要私有仓库
        )
        
        # 推送tokenizer
        tokenizer.push_to_hub(
            repo_id=hub_model_name,
            commit_message="Add tokenizer for Qwen2.5-tiny single layer model"
        )
        
        print(f"✅ 成功推送到 huggingface.co/{hub_model_name}")
        
    except Exception as e:
        print(f"❌ 推送失败: {str(e)}")
        print("请确保您已经登录 Hugging Face: huggingface-cli login")
        print("或设置了 HF_TOKEN 环境变量")

# 取消注释下面的行来推送到Hub
push_to_hub()