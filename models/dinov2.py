# ---INFO-----------------------------------------------------------------------
"""
Wrapper for DinoV2 models from https://github.com/facebookresearch/dinov2
"""

__all__ = [
    "DinoV2",
]


# ---CONSTANTS------------------------------------------------------------------
SUPPORTED_MODELS = [
    "dinov2_vits14_ld",
    "dinov2_vitb14_ld",
    "dinov2_vitl14_ld",
    "dinov2_vitg14_ld",
    "dinov2_vits14_dd",
    "dinov2_vitb14_dd",
    "dinov2_vitl14_dd",
    "dinov2_vitg14_dd",
]


# ---DEPENDENCIES---------------------------------------------------------------
import torch as pt


# ---DINOV2---------------------------------------------------------------------
class DinoV2:

    def __init__(self, dino: str = "dinov2_vits14_ld", cuda=True):
        if dino not in SUPPORTED_MODELS:
            raise ValueError(f"`model` must be one of {SUPPORTED_MODELS}")
        self.dino = pt.hub.load("facebookresearch/dinov2:main", dino)
        self.dino.eval()
        if cuda:
            self.dino.cuda()

    def encode(self, x: pt.Tensor) -> pt.Tensor:
        x = self.dino.backbone(x)
        return x

    def decode(self, x: pt.Tensor) -> pt.Tensor:
        x = self.dino.decode_head.forward_test(x, None)
        x = pt.clamp(
            x,
            min=self.dino.decode_head.min_depth,
            max=self.dino.decode_head.max_depth,
        )
        return x
