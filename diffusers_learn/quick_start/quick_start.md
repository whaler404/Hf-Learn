# stable-diffusion-v1-5/stable-diffusion-v1-5

## pipeline print
```json
StableDiffusionPipeline {
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.34.0",
  "_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "image_encoder": [
    null,
    null
  ],
  "requires_safety_checker": true,
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

## 源码阅读


stable diffusion 伪代码流程包括： unet 融合文本编码预测噪声，每个时间步向采样器传入 noise 、当前时间步 t 和 image ，最后用 vae 解码器解码图像
```python
def ldm_text_to_image(image_shape, text, ddim_steps = 20, eta = 0)
  ddim_scheduler = DDIMScheduler()
  vae = VAE()
  unet = UNet()
  zt = randn(image_shape)
  eta = input()
  T = 1000
  timesteps = ddim_scheduler.get_timesteps(T, ddim_steps) # [1000, 950, 900, ...]

  text_encoder = CLIP()
  c = text_encoder.encode(text)

  for t = timesteps:
    eps = unet(zt, t, c)
    std = ddim_scheduler.get_std(t, eta)
    zt = ddim_scheduler.get_xt_prev(zt, t, eps, std)
  xt = vae.decoder.decode(zt)
  return xt
```
通用架构：一般无需训练的编辑技术只需要修改调度器和注意力块，需要训练的编辑技术只要修改 unet 的块即可
VAE的解码和编码
文本编码器（CLIP）的编码
用U-Net预测当前图像应去除的噪声
用采样器计算下一去噪迭代的图像