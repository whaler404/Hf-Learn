import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from prepare_dataset import prepare_dataset
from model import TextGenerationModel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main():
    train_loader, val_loader, _ = prepare_dataset(num_workers=4)

    output_dir = "trainer_output/text_generation"
    os.makedirs(output_dir, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=f"{output_dir}/logs",
        name="text_generation"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=output_dir,
        filename="SmolLM2-135M-TinyStories-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        save_last=True,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        verbose=True
    )

    model = TextGenerationModel(learning_rate=5e-5)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    trainer.fit(model, train_loader, val_loader)

    # best_model_path = checkpoint_callback.best_model_path
    # print(f"Best model saved at: {best_model_path}")

    # model.save_model(f"{output_dir}/final_model")
    # print(f"Final model saved at: {output_dir}/final_model")


if __name__ == "__main__":
    main()