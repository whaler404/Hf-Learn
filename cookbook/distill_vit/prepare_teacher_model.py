import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torchmetrics import Accuracy

from prepare_dataset import train_loader, val_loader

torch.set_float32_matmul_precision('high')

class LitViT(pl.LightningModule):
    def __init__(self, model_name="google/vit-base-patch16-224", num_labels=3, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # 允许加载时略过分类头尺寸不匹配
        )

        # 冻结除分类头外的参数
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=num_labels)

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=-1)
        acc = self.acc(preds, batch["labels"])

        # Lightning 会自动在日志与进度条显示这些值
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)

        return {"val_loss": val_loss, "val_acc": acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

def main():

    model = LitViT()


    ckpt_cb = ModelCheckpoint(
        dirpath="trainer_output/distill_vit",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        filename="teacher-vit-{epoch:02d}-{val_acc:.4f}",
        auto_insert_metric_name=False,
    )
    es_cb = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=10,
    )
    lrmon = LearningRateMonitor(logging_interval="epoch")
    tb_logger = TensorBoardLogger(save_dir="trainer_output/distill_vit/logs", name="vit_teacher")

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=50,
        precision="16-mixed",  # 自动混合精度
        logger=tb_logger,
        callbacks=[ckpt_cb, es_cb, lrmon],
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
