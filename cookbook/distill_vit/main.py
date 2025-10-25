import os
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    ViTForImageClassification,
)

from datasets import load_dataset

# ----------------------------
# 1) 数据模块：ImageNet-1k
# ----------------------------
class HFDatasetWrapper(torch.utils.data.Dataset):
    """把 HuggingFace dataset 封装为 PyTorch Dataset."""
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]  # PIL.Image
        label = example["label"]
        enc = self.processor(image, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "datasets/imagenet-1k-subset/data",
        train_file: str = "imagenet_train.parquet",
        val_file: str = "imagenet_val.parquet",
        test_file: str = "imagenet_test.parquet",
        pretrained_processor_name: str = "google/vit-base-patch16-224",
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 使用与教师一致的预处理器
        self.processor = AutoImageProcessor.from_pretrained(pretrained_processor_name, use_fast=True)

    def setup(self, stage=None):
        # 加载 parquet 文件
        self.train_dataset = load_dataset(
            "parquet", data_files=f"{self.data_dir}/{self.train_file}"
        )["train"]
        self.val_dataset = load_dataset(
            "parquet", data_files=f"{self.data_dir}/{self.val_file}"
        )["train"]
        self.test_dataset = load_dataset(
            "parquet", data_files=f"{self.data_dir}/{self.test_file}"
        )["train"]

        # 转换为 PyTorch Dataset
        self.train_dataset = HFDatasetWrapper(self.train_dataset, self.processor)
        self.val_dataset = HFDatasetWrapper(self.val_dataset, self.processor)
        self.test_dataset = HFDatasetWrapper(self.test_dataset, self.processor)

        # 尝试获取类别数
        example = load_dataset("parquet", data_files=f"{self.data_dir}/{self.train_file}")["train"]
        if "label" in example.features and hasattr(example.features["label"], "num_classes"):
            self.num_classes = example.features["label"].num_classes
        else:
            # 若 label 为 int，自动推断最大类别数 + 1
            labels = [ex["label"] for ex in example.select(range(min(1000, len(example))))]
            self.num_classes = max(labels) + 1

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

# ------------------------------------------
# 2) 蒸馏 LightningModule：层对齐 + logits 蒸馏
# ------------------------------------------
class DistillViT(pl.LightningModule):
    def __init__(
        self,
        teacher_name_or_path: str = "google/vit-base-patch16-224",
        num_student_layers: int = 3,
        teacher_match_layers: List[int] = (3, 7, 11),  # 教师层（1-based, 与 HF hidden_states 对齐见下）
        student_match_layers: List[int] = (0, 1, 2),   # 学生层（1-based）
        temperature: float = 4.0,
        alpha_logits: float = 0.5,  # logits 蒸馏权重
        alpha_layers: float = 0.5,  # 层蒸馏权重
        lr: float = 5e-5,
        weight_decay: float = 0.05,
        num_classes: int = 1000,
        seed_from_teacher_prefix: bool = True,  # True：用教师前几层权重初始化学生
    ):
        super().__init__()
        self.save_hyperparameters()

        self.T = temperature
        self.alpha_logits = alpha_logits
        self.alpha_layers = alpha_layers
        self.lr = lr
        self.weight_decay = weight_decay

        # ---------- Teacher ----------
        t_config = AutoConfig.from_pretrained(teacher_name_or_path)
        # 确保输出 hidden_states
        t_config.output_hidden_states = True
        self.teacher = ViTForImageClassification.from_pretrained(
            teacher_name_or_path,
            config=t_config,
        )
        # 冻结教师
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # ---------- Student ----------
        s_config = AutoConfig.from_pretrained(teacher_name_or_path)
        s_config.num_hidden_layers = num_student_layers
        s_config.num_labels = num_classes
        s_config.output_hidden_states = True
        self.student = ViTForImageClassification(s_config)

        with torch.no_grad():
            # Embedding / Norm / Classifier 直接复制
            self.student.vit.embeddings.load_state_dict(self.teacher.vit.embeddings.state_dict(), strict=False)
            self.student.vit.layernorm.load_state_dict(self.teacher.vit.layernorm.state_dict(), strict=False)
            self.student.classifier.load_state_dict(self.teacher.classifier.state_dict(), strict=False)

            # ---------- 匹配层复制 ----------
            t_blocks = self.teacher.vit.encoder.layer
            s_blocks = self.student.vit.encoder.layer

            teacher_map = {0: 3, 1: 7, 2: 11}  # 学生->教师 层映射
            for s_idx, t_idx in teacher_map.items():
                s_blocks[s_idx].load_state_dict(t_blocks[t_idx].state_dict(), strict=False)

            print("✅ 学生模型初始化完成：")
            for s_idx, t_idx in teacher_map.items():
                print(f"  学生第 {s_idx} 层 ← 教师第 {t_idx} 层")

        # 记录层对齐（使用 HF 的 hidden_states 规则：长度 = num_layers+1，index 0 是 embeddings 后输出，
        # index k 表示第 k 层后的输出；因此“第 L 层”对应 hidden_states[L]）
        self.t_layers = list(teacher_match_layers)
        self.s_layers = list(student_match_layers)

        # 分类损失
        self.ce = nn.CrossEntropyLoss()

    @staticmethod
    def _kl_div_with_temperature(student_logits, teacher_logits, T: float):
        # KL(student || teacher) with temperature scaling
        return F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

    @staticmethod
    def _layer_kl(student_hidden, teacher_hidden, T: float):
        """
        对齐每一层的隐藏表征：取 CLS token（[:, 0, :]），
        沿最后一维做 softmax 得到分布，然后做 KL。
        """
        s_cls = student_hidden[:, 0, :]  # [B, D]
        t_cls = teacher_hidden[:, 0, :]  # [B, D]
        return F.kl_div(
            F.log_softmax(s_cls / T, dim=-1),
            F.softmax(t_cls / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

    def forward(self, pixel_values):
        return self.student(pixel_values=pixel_values)

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch["pixel_values"], batch["labels"]

        with torch.no_grad():
            t_out = self.teacher(pixel_values=pixel_values, output_hidden_states=True)

        s_out = self.student(pixel_values=pixel_values, output_hidden_states=True)

        # 1) 硬标签交叉熵
        hard_loss = self.ce(s_out.logits, labels)

        # 2) logits 蒸馏（soft）
        logits_kd = self._kl_div_with_temperature(s_out.logits, t_out.logits, self.T)

        # 3) 中间层蒸馏（soft）：学生[1,2,3] 对齐 教师[4,8,12]
        layer_kds = []
        for sL, tL in zip(self.s_layers, self.t_layers):
            s_h = s_out.hidden_states[sL]   # 注意：hidden_states[1] 即第1层后的输出
            t_h = t_out.hidden_states[tL]
            layer_kds.append(self._layer_kl(s_h, t_h, self.T))
        layers_kd = torch.stack(layer_kds).mean()

        # 总损失
        loss = (1 - self.alpha_logits - self.alpha_layers) * hard_loss \
               + self.alpha_logits * logits_kd \
               + self.alpha_layers * layers_kd

        # 日志
        self.log_dict({
            "train/hard_ce": hard_loss,
            "train/kd_logits": logits_kd,
            "train/kd_layers": layers_kd,
            "train/loss": loss,
        }, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch["pixel_values"], batch["labels"]
        with torch.no_grad():
            s_out = self.student(pixel_values=pixel_values)
        val_loss = self.ce(s_out.logits, labels)

        # top-1 acc
        preds = s_out.logits.argmax(dim=-1)
        acc1 = (preds == labels).float().mean()

        self.log_dict({
            "val/loss": val_loss,
            "val/acc1": acc1,
        }, prog_bar=True, on_epoch=True, sync_dist=True)

        return {"val_loss": val_loss, "acc1": acc1}

    def configure_optimizers(self):
        # 典型的 ViT 优化设置
        decay, no_decay = [], []
        for n, p in self.student.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n for nd in ["bias", "LayerNorm.weight"]):
                no_decay.append(p)
            else:
                decay.append(p)
        optim = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(0.9, 0.999),
        )
        # 余弦退火 + 线性 warmup（简化：仅余弦）
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.trainer.max_epochs)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ----------------------------
# 3) 训练脚手架
# ----------------------------
def main():
    pl.seed_everything(42)

    # 使用类的默认值
    # IMAGENET_ROOT = "datasets/imagenet-1k-subset/data"  
    # BATCH_SIZE = 64 
    # NUM_WORKERS = 8 
    # MAX_EPOCHS = 50
    # ACCELERATOR = "gpu" if torch.cuda.is_available() else "auto"
    # DEVICES = "auto"

    IMAGENET_ROOT = "datasets/imagenet-1k-subset-tiny/data"  
    BATCH_SIZE = 16
    NUM_WORKERS = 8 
    MAX_EPOCHS = 10
    ACCELERATOR = "gpu" if torch.cuda.is_available() else "auto"
    DEVICES = "auto"

    dm = ImageNetDataModule(
        data_dir=IMAGENET_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    dm.setup()

    model = DistillViT(num_classes=dm.num_classes)

    ckpt_cb = ModelCheckpoint(
        dirpath="trainer_output/distill_vit",
        monitor="val/acc1",
        mode="max",
        save_top_k=3,
        filename="student-vit-{epoch:02d}-{val_acc1:.4f}",
        auto_insert_metric_name=False,
    )
    es_cb = EarlyStopping(
        monitor="val/acc1",
        mode="max",
        patience=10,
    )
    lrmon = LearningRateMonitor(logging_interval="epoch")
    tb_logger = TensorBoardLogger(save_dir="trainer_output/distill_vit/logs", name="vit_kd")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        logger=tb_logger,
        callbacks=[ckpt_cb, es_cb, lrmon],
        deterministic=False,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
