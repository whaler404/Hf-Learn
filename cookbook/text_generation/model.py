import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PrefixTuningConfig, TaskType
import pytorch_lightning as pl
from torchmetrics import MeanMetric

class TextGenerationModel(pl.LightningModule):
    def __init__(self, model_name="HuggingFaceTB/SmolLM2-135M", learning_rate=5e-5):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )

        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            prefix_projection=True
        )

        self.model = get_peft_model(self.model, peft_config)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.train_loss.update(loss)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.val_loss.update(loss)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)