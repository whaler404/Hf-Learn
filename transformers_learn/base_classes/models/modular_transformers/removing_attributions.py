import torch.nn as nn

class DummyConfig:
  def __init__(self, attribute=None):
    self.attribute = attribute
    
class MyNewDummyConfig(DummyConfig):
  def __init__(self, attribute=None):
    super().__init__(attribute)
    # Additional attributes or modifications can be added here

class DummyModel(nn.Module):

  def __init__(self, config: DummyConfig):
    super().__init__()
    self.attribute = config.attribute
    if self.attribute:
      # do more stuff with `self.attribute` here
      ...

class MyNewDummyModel(DummyModel):

  def __init__(self, config: MyNewDummyConfig):
    super().__init__(config)
    del self.attribute