"""
dinosiglip_vit.py

Vision backbone that returns concatenated features from both DINOv2 and SigLIP.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

import timm
import torch
from PIL import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize
from torch.distributed.fsdp.wrap import _module_wrap_policy, transformer_auto_wrap_policy
from prismatic.models.backbones.vision.base_vision import (
    ImageTransform,
    LetterboxPad,
    VisionBackbone,
    compute_sequence_patches,
    unpack_tuple,
)

# Registry =>> Supported DinoSigLIP Pairs (as TIMM identifiers)
DINOSigLIP_VISION_BACKBONES = {
    "dinosiglip-vit-so-224px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_224",
    },
    "dinosiglip-vit-so-384px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_384",
    },
}


@dataclass
class DinoSigLIPImageTransform:
    dino_image_transform: ImageTransform
    siglip_image_transform: ImageTransform
    is_prismatic: bool = True

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        return {"dino": self.dino_image_transform(img, **kwargs), "siglip": self.siglip_image_transform(img, **kwargs)}


class DinoSigLIPViTBackbone(VisionBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        image_sequence_len: int = 1,
    ) -> None:
        super().__init__(
            vision_backbone_id,
            image_resize_strategy,
            default_image_size=default_image_size,
            image_sequence_len=image_sequence_len,
        )
        self.dino_timm_path_or_url = DINOSigLIP_VISION_BACKBONES[vision_backbone_id]["dino"]
        self.siglip_timm_path_or_url = DINOSigLIP_VISION_BACKBONES[vision_backbone_id]["siglip"]

        # Initialize both Featurizers (ViTs) by downloading from HF / TIMM Hub if necessary
        # DINOv2 with pretrained weights
        self.dino_featurizer: VisionTransformer = timm.create_model(
            self.dino_timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
        )
        self.dino_featurizer.eval()
        
        # SigLIP with pretrained weights
        self.siglip_featurizer: VisionTransformer = timm.create_model(
            self.siglip_timm_path_or_url, pretrained=True, num_classes=0, img_size=self.default_image_size
        )
        self.siglip_featurizer.eval()

        # Setup data transforms for both models
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        self.siglip_data_cfg = timm.data.resolve_model_data_config(self.siglip_featurizer)

        # Initialize *both* Transforms
        default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        default_siglip_transform = timm.data.create_transform(**self.siglip_data_cfg, is_training=False)
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_dino_transform, Compose), "Unexpected `default_dino_image_transform`!"
            assert isinstance(default_siglip_transform, Compose), "Unexpected `default_siglip_image_transform`!"
            assert isinstance(default_dino_transform.transforms[0], Resize)
            assert isinstance(default_siglip_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            dino_transform = Compose(
                [
                    Resize(target_size, interpolation=default_dino_transform.transforms[0].interpolation),
                    *default_dino_transform.transforms[1:],
                ]
            )
            siglip_transform = Compose(
                [
                    Resize(target_size, interpolation=default_siglip_transform.transforms[0].interpolation),
                    *default_siglip_transform.transforms[1:],
                ]
            )

            self.image_transform = DinoSigLIPImageTransform(dino_transform, siglip_transform)

        elif self.image_resize_strategy == "letterbox":
            # Both DINOv2 and SigLIP have 14-patch convolutions, so only need to specify `letterbox_pad` for the first
            self.image_transform = DinoSigLIPImageTransform(
                Compose([LetterboxPad(self.dino_data_cfg["input_size"][1]), default_dino_transform]),
                Compose([LetterboxPad(self.siglip_data_cfg["input_size"][1]), default_siglip_transform]),
            )
        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_image_transform(self) -> DinoSigLIPImageTransform:
        return self.image_transform

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Runs the transformed image/pixel tensors through each vision backbone, returning concatenated patches."""

        if self.image_sequence_len == 1:
            dino_patches = self.dino_featurizer(pixel_values["dino"])
            siglip_patches = self.siglip_featurizer(pixel_values["siglip"])

        else:
            featurizers = {
                "dino": self.dino_featurizer,
                "siglip": self.siglip_featurizer,
            }
            patches = compute_sequence_patches(pixel_values, featurizers, self.image_sequence_len)
            dino_patches, siglip_patches = patches["dino"], patches["siglip"]
        #return torch.cat([dino_patches, siglip_patches], dim=2)原本

        return torch.cat([dino_patches, siglip_patches], dim=-1)#因为只有两个维度

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        assert self.dino_featurizer.patch_embed.num_patches == self.siglip_featurizer.patch_embed.num_patches
        return self.dino_featurizer.patch_embed.num_patches * self.image_sequence_len

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
    @property
    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy for this vision backbone."""

        # Get transformer blocks from both DINO and SigLIP models
        transformer_blocks = []
        if hasattr(self.dino, "blocks"):
            transformer_blocks.extend(list(self.dino.blocks))
        if hasattr(self.siglip, "blocks"):
            transformer_blocks.extend(list(self.siglip.blocks))

        # Create wrapping policy
        policies = [_module_wrap_policy]
        if transformer_blocks:
            policies.append(partial(transformer_auto_wrap_policy, transformer_layer_cls=set(transformer_blocks)))

        def lambda_policy_fn(module):
            # Custom policy for specific modules if needed
            return False

        policies.append(lambda_policy_fn)

        return partial(_or_policy, policies=policies)

