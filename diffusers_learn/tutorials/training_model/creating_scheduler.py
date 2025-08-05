import torch
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
timesteps = torch.LongTensor([50])