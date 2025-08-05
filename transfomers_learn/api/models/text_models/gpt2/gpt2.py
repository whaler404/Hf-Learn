from transformers.models.gpt2.modeling_gpt2 import GPT2ForTokenClassification
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

# tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
# model = GPT2ForTokenClassification.from_pretrained("openai-community/gpt2")

from transformers import AutoConfig
config = AutoConfig.from_pretrained("./transfomers_learn/api/models/text_models/gpt2/config.json")
from transformers import AutoModel
model = AutoModel.from_config(config)

print(model)

from torchinfo import summary
summary(model)
