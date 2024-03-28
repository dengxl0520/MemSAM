# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from turtle import shape
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .mem import Mem
from einops import rearrange


class MemSAM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        memory: Mem,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.memory = memory
        if self.memory is not None:
            self.memory = memory
            self.memory.key_encoder = image_encoder

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        # for param in self.image_encoder.parameters():
        #   param.requires_grad = False
        for n, value in self.image_encoder.named_parameters():
            if "cnn_embed" not in n and "post_pos_embed" not in n and "Adapter" not in n and "2.attn.rel_pos" not in n and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n and "upneck" not in n:
                value.requires_grad = False
        pass
        
    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward_sam(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def forward(
        self,
        imgs: torch.Tensor, # [b,t,c,h,w]
        pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
        bbox: torch.Tensor=None, # b 4
    ) -> torch.Tensor:
        if self.memory is not None:
            return self._forward_with_memory(imgs,pt,bbox)
        else:
            return self._forward_without_memory(imgs,pt,bbox)

    def _forward_without_memory(
        self,
        imgs: torch.Tensor, # [b,t,c,h,w]
        pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
        bbox: torch.Tensor=None, # b 4
    ) -> torch.Tensor:
        b, t, c, h, w = imgs.shape  # b t c h w
        imgs = rearrange(imgs, "b t c h w -> (b t) c h w")
        imgs= self.image_encoder(imgs)
        imgs = rearrange(imgs, "(b t) c h w -> b t c h w", b=b)
        frames_pred = []
        for ti in range(0, t):
            frame = imgs[:,ti,:,:,:]
            se, de = self.prompt_encoder(# se b 2 256, de b 256 32 32
                        points=(pt[0][:,0],pt[1][:1]),
                        boxes=None,
                        masks=None,
                    )
            mask, _ = self.mask_decoder( # low_res_mask b 1 128 128
                        image_embeddings=frame,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    ) # b c h w
            mask = F.interpolate(mask, (h,w), mode="bilinear", align_corners=False) #b 1 256 256
            frames_pred.append(mask)
        pred = torch.stack(frames_pred, dim=1) # b t c h w

        return pred

    def _forward_with_memory(
        self,
        imgs: torch.Tensor, # [b,t,c,h,w]
        pt: Tuple[torch.Tensor, torch.Tensor],  # ([b n 1 2], [b n])
        bbox: torch.Tensor=None, # b 4
    ) -> torch.Tensor:
        b, t, c, h, w = imgs.shape  # b t c h w
        # encode imgs to imgs embedding
        key, shrinkage, selection, imge = self.memory('encode_key', imgs)
        # init memory
        hidden = torch.zeros((b, 1, self.memory.hidden_dim, *key.shape[-2:])).to(imge.device)
        frames_pred = []
        # first frame
        if pt is not None:
            se, de = self.prompt_encoder(# se b 2 256, de b 256 32 32
                        points=(pt[0][:,0],pt[1][:1]),
                        boxes=None,
                        masks=None,
                    )
        else:
            se, de = self.prompt_encoder(# se b 2 256, de b 256 32 32
                        points=None,
                        boxes=None,
                        masks=None,
                    )
        mask, _ = self.mask_decoder(
                    image_embeddings=imge[:,0],
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                ) # b c h w
        mask = F.interpolate(mask, imgs.shape[-2:], mode="bilinear", align_corners=False) #b 1 256 256
        # frames_pred.append(mask)
        values_0, hidden = self.memory('encode_value', imgs[:,0], imge[:,0], hidden, mask)
        values = values_0[:,:,:,:0]

        # process frames
        for ti in range(0, t):
            if ti == 0 :
                ref_keys = key[:,:,[0]] 
                ref_shrinkage = shrinkage[:,:,[0]]
                ref_values = values_0 
            else:
                ref_keys = key[:,:,:ti]
                ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
                ref_values = values

            # get single frame
            frame = imge[:,ti]
            # read memory
            memory_readout = self.memory(
                'read_memory',
                key[:, :, ti],
                selection[:, :, ti] if selection is not None else None,
                ref_keys, ref_shrinkage, ref_values)
            # generate memory embedding
            hidden, me = self.memory('decode', frame, hidden, memory_readout)
            # # featmap
            # from mmengine.visualization import Visualizer
            # visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')],
            #                         save_dir='temp_dir')
            # drawn_img = visualizer.draw_featmap(featmap=me[0,0]*-1,
            #                         overlaid_image=imgs[0,ti].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8),
            #                         channel_reduction='squeeze_mean',
            #                         alpha=0.3)
            # if self.memory.reinforce :
            #     visualizer.add_image(f'featmap_reinforce', drawn_img, step=ti)
            # else:
            #     visualizer.add_image(f'featmap_noreinforce', drawn_img, step=ti)

            mask, _ = self.mask_decoder( 
                        image_embeddings=frame,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=None,
                        dense_prompt_embeddings=me[:,0], # remove object num dim
                        multimask_output=False,
                    ) # b c h w
            mask = F.interpolate(mask, imgs.shape[-2:], mode="bilinear", align_corners=False) #b 1 256 256
            frames_pred.append(mask)

            # last frame no encode
            if ti < t-1:
                # update memory
                is_deep_update = np.random.rand() < 0.2
                # v16, hidden = self.memory('encode_value', imgs[:,ti], me[:,0], hidden, mask, is_deep_update=is_deep_update)
                v16, hidden = self.memory('encode_value', imgs[:,ti], imge[:,ti], hidden, mask, is_deep_update=is_deep_update)
                values = torch.cat([values, v16], 3)

        pred = torch.stack(frames_pred, dim=1) # b t c h w

        return pred

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
