# Load model directly and modify architecture
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy

# 加载原始模型和tokenizer
print("正在加载原始模型...")
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
original_model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

print(f"原始模型层数: {original_model.config.num_hidden_layers}")

# 创建新的配置，只保留1层
new_config = copy.deepcopy(original_model.config)
new_config.num_hidden_layers = 1
new_config.max_window_layers = 1  # 同时调整相关参数

print(f"新模型层数: {new_config.num_hidden_layers}")

# 使用新配置创建模型
print("正在创建修改后的模型...")
model = Qwen2ForCausalLM(new_config)

# 复制权重 - 只复制第一层
print("正在复制权重...")
with torch.no_grad():
    # 复制embedding层
    model.model.embed_tokens.weight.copy_(original_model.model.embed_tokens.weight)
    
    # 只复制第一层的权重
    model.model.layers[0].load_state_dict(original_model.model.layers[0].state_dict())
    
    # 复制norm层和lm_head
    model.model.norm.weight.copy_(original_model.model.norm.weight)
    model.lm_head.weight.copy_(original_model.lm_head.weight)

print("权重复制完成！")

# 测试模型
print("\n正在测试修改后的模型...")
test_input = "Hello, how are you?"
inputs = tokenizer(test_input, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成的文本: {generated_text}")

print(f"\n模型参数总数: {model.num_parameters():,}")
print(f"原始模型参数总数: {original_model.num_parameters():,}")

# 保存修改后的模型到本地
local_save_path = "./trainer_output/qwen2-0.5b-single-layer"
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
        
        # 设置Hub仓库名称（请根据需要修改）
        hub_model_name = "qwen2-0.5b-single-layer"  # 修改为您想要的名称
        
        # 推送模型
        model.push_to_hub(
            repo_id=hub_model_name,
            commit_message="Add Qwen2.5-0.5B model with single hidden layer",
            private=False  # 设置为True如果您想要私有仓库
        )
        
        # 推送tokenizer
        tokenizer.push_to_hub(
            repo_id=hub_model_name,
            commit_message="Add tokenizer for Qwen2.5-0.5B single layer model"
        )
        
        print(f"✅ 成功推送到 huggingface.co/{hub_model_name}")
        
    except Exception as e:
        print(f"❌ 推送失败: {str(e)}")
        print("请确保您已经登录 Hugging Face: huggingface-cli login")
        print("或设置了 HF_TOKEN 环境变量")

# 取消注释下面的行来推送到Hub
# push_to_hub()

# {
#   "architectures": [
#     "Qwen2ForCausalLM"
#   ],
#   "attention_dropout": 0.0,
#   "bos_token_id": 151643,
#   "eos_token_id": 151643,
#   "hidden_act": "silu",
#   "hidden_size": 896,
#   "initializer_range": 0.02,
#   "intermediate_size": 4864,
#   "max_position_embeddings": 32768,
#   "max_window_layers": 24,
#   "model_type": "qwen2",
#   "num_attention_heads": 14,
#   "num_hidden_layers": 24,
#   "num_key_value_heads": 2,
#   "rms_norm_eps": 1e-06,
#   "rope_theta": 1000000.0,
#   "sliding_window": 32768,
#   "tie_word_embeddings": true,
#   "torch_dtype": "bfloat16",
#   "transformers_version": "4.40.1",
#   "use_cache": true,
#   "use_mrope": false,
#   "use_sliding_window": false,
#   "vocab_size": 151936
# }