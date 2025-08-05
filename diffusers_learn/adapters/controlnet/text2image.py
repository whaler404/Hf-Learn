from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionControlNetPipeline

unet = UNet2DConditionModel.from_pretrained(
    "hf-internal-testing/tiny-stable-diffusion-torch",
    subfolder="unet",
    # torch_dtype=None,  # Use default dtype
)

controlnet = ControlNetModel.from_unet(
    unet,
)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "hf-internal-testing/tiny-stable-diffusion-torch", controlnet=controlnet, #torch_dtype=torch.float16
)

prompt = """
A photorealistic overhead image of a cat reclining sideways in a flamingo pool floatie holding a margarita. 
The cat is floating leisurely in the pool and completely relaxed and happy.
"""

import torch
canny_image = torch.randn(1, 3, 64, 64)  # Placeholder for canny image tensor

image = pipeline(
    prompt, 
    control_image=canny_image,
    controlnet_conditioning_scale=0.5,
    num_inference_steps=50, 
    guidance_scale=3.5,
).images[0]

print(image.size)  # Output the size of the generated image

print(controlnet)