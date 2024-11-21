from typing import Any, Dict, Union

import torch
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
import cv2

from utils.dataset_configuration import resize_max_res_tensor
from models.unet_2d_condition import UNet2DConditionModel
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer
from dataloader.flash_loader import *
from utils.image_util import chw2hwc
from utils.normal_ensemble import ensemble_normals


class NormalPipelineOutput(BaseOutput):
    normal_np: np.ndarray
    normal_colored: Image.Image


class NormalEstimationPipeline(DiffusionPipeline):
    # two hyper-parameters
    latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        test_dir: str='',
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        domain: str = "object",
        show_progress_bar: bool = True,
    ) -> NormalPipelineOutput:

        # inherit from thea Diffusion Pipeline
        device = self.device

        # adjust the input resolution.
        if not match_input_res:
            assert (
                processing_res is not None
            ), " Value Error: `resize_output_back` is only valid with "

        assert processing_res >= 0
        assert denoising_steps >= 1
        assert ensemble_size >= 1

        # ----------------- predicting normal -----------------
        normal_pred_ls = []

        iterable_bar = tqdm(
            range(ensemble_size), desc=" " * 2 + "Inference batches", leave=False
        )

        for batch in iterable_bar:
            normal_pred_raw,r_s,r_e,c_s,c_e = self.single_infer(
                test_dir=test_dir,
                num_inference_steps=denoising_steps,
                domain=domain,
                show_pbar=show_progress_bar,
            )
            normal_pred_ls.append(normal_pred_raw.detach().clone())

        normal_preds = torch.concat(normal_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            normal_pred = ensemble_normals(normal_preds)
        else:
            normal_pred = normal_preds

        # ----------------- Post processing -----------------
        normal_pred = normal_pred.cpu().numpy().astype(np.float32)

        normal_pred = cv2.resize(
            chw2hwc(normal_pred), (512, 512), interpolation=cv2.INTER_NEAREST
        )
        normal_pred[:, :, 0] = -normal_pred[:, :, 0]
        # Clip output range: current size is the original size
        normal_pred = normal_pred.clip(-1, 1)

        normal_colored = ((normal_pred + 1) / 2 * 255).astype(np.uint8)
        nml = cv2.resize(normal_colored, dsize=(c_e - c_s, r_e - r_s), interpolation=cv2.INTER_CUBIC)
        nout = np.zeros((512, 512, 3), np.float32)
        nout[r_s:r_e,c_s:c_e, :] = nml

        return NormalPipelineOutput(
            normal_np = normal_pred,
            normal_colored = nout,
        )

    def __encode_text(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)  # [1,2]
        text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)  # [1,2,1024]
        return text_embed

    @torch.no_grad()
    def single_infer(
        self,
        test_dir: str,
        num_inference_steps: int,
        domain: str,
        show_pbar: bool,
    ):

        device = 'cuda'

        # Set timesteps: inherit from the diffuison pipeline
        self.scheduler.set_timesteps(
            num_inference_steps, device=device
        )  # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]

        i1, i2, r_s, r_e, c_s, c_e = load_eval(test_dir)
        image1 = i1.transpose(2, 0, 1)
        image2 = i2.transpose(2, 0, 1)
        image1 = np.expand_dims(image1, axis=0)
        image2 = np.expand_dims(image2, axis=0)
        image1 = torch.from_numpy(image1).to("cuda", torch.float32)
        image2 = torch.from_numpy(image2).to("cuda", torch.float32)
        image_data_resized1 = resize_max_res_tensor(image1, mode="rgb")
        image_data_resized2 = resize_max_res_tensor(image2, mode="rgb")
        rgb_latent1 = self.encode_RGB(image_data_resized1)
        rgb_latent2 = self.encode_RGB(image_data_resized2)
        rgb_latent = torch.cat((rgb_latent1, rgb_latent2), dim=1)

        # Initial geometric maps (Guassian noise)
        geo_latent = torch.randn(rgb_latent1.shape, device=device, dtype=self.dtype)

        batch_text_embeds = self.__encode_text("object geometry").repeat(
            (rgb_latent.shape[0], 1, 1)
        )

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, geo_latent], dim=1)

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_text_embeds
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            geo_latent = self.scheduler.step(noise_pred, t, geo_latent).prev_sample

        geo_latent = geo_latent
        torch.cuda.empty_cache()

        normal = self.decode_normal(geo_latent)
        normal /= torch.norm(normal, p=2, dim=1, keepdim=True) + 1e-5
        normal *= -1.0

        return normal,r_s,r_e,c_s,c_e

    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.
        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.
        Returns:
            `torch.Tensor`: Image latent.
        """

        # encode
        h = self.vae.encoder(rgb_in)

        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.latent_scale_factor

        return rgb_latent

    def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode normal latent into normal map.
        Args:
            normal_latent (`torch.Tensor`):
        Returns:
            `torch.Tensor`: Decoded normal map.
        """

        # scale latent
        normal_latent = normal_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(normal_latent)
        normal = self.vae.decoder(z)
        return normal
