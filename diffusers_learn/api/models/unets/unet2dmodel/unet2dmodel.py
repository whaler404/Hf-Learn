from diffusers.models.unets.unet_2d import UNet2DModel

config_path = './diffusers_learn/api/models/unets/unet2dmodel/unet2dmodel.json'
import json
with open(config_path, 'r') as f:
    config = f.read()
    config = json.loads(config)

import torch
from diffusers.models.unets.unet_2d import UNet2DOutput, UNet2DModel, register_to_config
from typing import Optional, Union, Tuple
class MyUNet2DModel(UNet2DModel):
    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        mid_block_type: Optional[str] = "UNetMidBlock2D",
        up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            freq_shift=freq_shift,
            flip_sin_to_cos=flip_sin_to_cos,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            mid_block_scale_factor=mid_block_scale_factor,
            downsample_padding=downsample_padding,
            downsample_type=downsample_type,
            upsample_type=upsample_type,
            dropout=dropout,
            act_fn=act_fn,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            attn_norm_num_groups=attn_norm_num_groups,
            norm_eps=norm_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=add_attention,
            class_embed_type=class_embed_type,
            num_class_embeds=num_class_embeds,
            num_train_timesteps=num_train_timesteps
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0
        print(f"[0] sample(centered): {sample.shape}")

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        print(f"[1] timesteps: {timesteps.shape}")

        t_emb = self.time_proj(timesteps)
        print(f"[1] t_emb: {t_emb.shape}")

        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        print(f"[1] emb: {emb.shape}")

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            print(f"[1] class_emb: {class_emb.shape}")
            emb = emb + class_emb
            print(f"[1] emb+class_emb: {emb.shape}")
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)
        print(f"[2] sample after conv_in: {sample.shape}")

        # 3. down
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            print(f"[3] down block {i} sample: {sample.shape}")
            if isinstance(res_samples, (list, tuple)):
                for j, res in enumerate(res_samples):
                    print(f"[3] down block {i} res_samples[{j}]: {res.shape}")
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)
            print(f"[4] mid_block sample: {sample.shape}")

        # 5. up
        skip_sample = None
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)
            print(f"[5] up block {i} sample: {sample.shape}")

        # 6. post-process
        sample = self.conv_norm_out(sample)
        print(f"[6] after conv_norm_out: {sample.shape}")
        sample = self.conv_act(sample)
        print(f"[6] after conv_act: {sample.shape}")
        sample = self.conv_out(sample)
        print(f"[6] after conv_out: {sample.shape}")

        if skip_sample is not None:
            sample += skip_sample
            print(f"[6] after skip_sample add: {sample.shape}")

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps
            print(f"[6] after fourier time embedding: {sample.shape}")

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)

model = MyUNet2DModel.from_config(config)

images = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)

model_output = model(sample=images, timestep=2)

# [0] sample(centered): torch.Size([1, 3, 256, 256])
# [1] timesteps: torch.Size([1])
# [1] t_emb: torch.Size([1, 128])
# [1] emb: torch.Size([1, 512])
# [2] sample after conv_in: torch.Size([1, 128, 256, 256])
# [3] down block 0 sample: torch.Size([1, 128, 128, 128])
# [3] down block 0 res_samples[0]: torch.Size([1, 128, 256, 256])
# [3] down block 0 res_samples[1]: torch.Size([1, 128, 128, 128])
# [3] down block 1 sample: torch.Size([1, 128, 64, 64])
# [3] down block 1 res_samples[0]: torch.Size([1, 128, 128, 128])
# [3] down block 1 res_samples[1]: torch.Size([1, 128, 64, 64])
# [3] down block 2 sample: torch.Size([1, 256, 32, 32])
# [3] down block 2 res_samples[0]: torch.Size([1, 256, 64, 64])
# [3] down block 2 res_samples[1]: torch.Size([1, 256, 32, 32])
# [3] down block 3 sample: torch.Size([1, 256, 16, 16])
# [3] down block 3 res_samples[0]: torch.Size([1, 256, 32, 32])
# [3] down block 3 res_samples[1]: torch.Size([1, 256, 16, 16])
# [3] down block 4 sample: torch.Size([1, 512, 8, 8])
# [3] down block 4 res_samples[0]: torch.Size([1, 512, 16, 16])
# [3] down block 4 res_samples[1]: torch.Size([1, 512, 8, 8])
# [3] down block 5 sample: torch.Size([1, 512, 8, 8])
# [3] down block 5 res_samples[0]: torch.Size([1, 512, 8, 8])
# [4] mid_block sample: torch.Size([1, 512, 8, 8])
# [5] up block 0 sample: torch.Size([1, 512, 16, 16])
# [5] up block 1 sample: torch.Size([1, 512, 32, 32])
# [5] up block 2 sample: torch.Size([1, 256, 64, 64])
# [5] up block 3 sample: torch.Size([1, 256, 128, 128])
# [5] up block 4 sample: torch.Size([1, 128, 256, 256])
# [5] up block 5 sample: torch.Size([1, 128, 256, 256])
# [6] after conv_norm_out: torch.Size([1, 128, 256, 256])
# [6] after conv_act: torch.Size([1, 128, 256, 256])
# [6] after conv_out: torch.Size([1, 3, 256, 256])

# print(model)