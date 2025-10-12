from transformers import AutoConfig

my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)

from transformers import AutoModel

my_model = AutoModel.from_config(my_config)