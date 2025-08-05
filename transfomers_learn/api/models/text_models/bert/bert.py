from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")
model = BertForMaskedLM.from_pretrained("google-bert/bert-base-cased")

print(model)

from torchinfo import summary
summary(model)