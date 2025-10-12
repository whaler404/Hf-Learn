from torch import nn
from transformers.models import BertConfig, BertModel, BertEmbeddings, BertForMaskedLM

# RoBERTa and BERT config is identical
class RobertaConfig(BertConfig):
  model_type = 'roberta'

# Redefine the embeddings to highlight the padding id difference, and redefine the position embeddings
class RobertaEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config())

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

# RoBERTa and BERT model is identical except for the embedding layer, which is defined above, so no need for additional changes here
class RobertaModel(BertModel):
  def __init__(self, config):
    super().__init__(config)
    self.embeddings = RobertaEmbeddings(config)


# The model heads now only need to redefine the model inside to `RobertaModel`
class RobertaForMaskedLM(BertForMaskedLM):
  def __init__(self, config):
    super().__init__(config)
    self.model = RobertaModel(config)