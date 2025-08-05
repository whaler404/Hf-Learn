import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from training_configuration import config
from loading_dataset import train_dataloader
from creating_unet2d_model import model
from creating_scheduler import noise_scheduler

from diffusers.optimization import get_cosine_schedule_with_warmup

import torch
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

import torch.nn.functional as F
model = model.to("cuda:0")
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to("cuda:0")
            print("clean_images shape:", clean_images.shape)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device="cuda:0")
            print("noise shape:", noise.shape)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image 随机采样噪声添加的步数
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device="cuda:0",
                dtype=torch.int64
            )
            print("timesteps shape:", timesteps.shape)

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process) 前向加噪，传入干净图像、噪声和步数
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            print("noisy_images shape:", noisy_images.shape)

            with accelerator.accumulate(model):
                # Predict the noise residual 加噪后的图像传入模型，结合时间步预测噪声
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                print("noise_pred shape:", noise_pred.shape)
                print("noise (target) shape:", noise.shape)
                # 计算预测噪声和真实噪声之间的均方误差损失
                loss = F.mse_loss(noise_pred, noise)
                print("loss shape:", loss.shape)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # clean_images shape: torch.Size([4, 3, 128, 128])
            # noise shape: torch.Size([4, 3, 128, 128])
            # timesteps shape: torch.Size([4])
            # noisy_images shape: torch.Size([4, 3, 128, 128])
            # noise_pred shape: torch.Size([4, 3, 128, 128])
            # noise (target) shape: torch.Size([4, 3, 128, 128])
            # loss shape: torch.Size([])
            break
        break

        # # After each epoch you optionally sample some demo images with evaluate() and save the model
        # if accelerator.is_main_process:
        #     pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

        #     # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
        #     #     evaluate(config, epoch, pipeline)

        #     if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
        #         if config.push_to_hub:
        #             upload_folder(
        #                 repo_id=repo_id,
        #                 folder_path=config.output_dir,
        #                 commit_message=f"Epoch {epoch}",
        #                 ignore_patterns=["step_*", "epoch_*"],
        #             )
        #         else:
        #             pipeline.save_pretrained(config.output_dir)

from accelerate import notebook_launcher

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)