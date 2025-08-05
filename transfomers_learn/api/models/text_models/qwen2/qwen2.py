# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

print(model)

from torchinfo import summary
summary(model)