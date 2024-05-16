# ---INFO-----------------------------------------------------------------------
"""
Wrapper for DinoV2 models from https://github.com/facebookresearch/dinov2
"""

__all__ = [
    "DinoV2",
]


# ---DEPENDENCIES---------------------------------------------------------------
import torch as pt


# ---DINOV2---------------------------------------------------------------------
class DinoV2:
    """
    DinoV2 model wrapper.

    Args:
    ----
        - dino (str): DinoV2 model name.
        - mode (str): Call mode.
        - cuda (bool): Use CUDA.
    """

    SUPPORTED_DINOS = [
        "dinov2_vits14_ld",
        "dinov2_vitb14_ld",
        "dinov2_vitl14_ld",
        "dinov2_vitg14_ld",
        "dinov2_vits14_dd",
        "dinov2_vitb14_dd",
        "dinov2_vitl14_dd",
        "dinov2_vitg14_dd",
    ]
    CALL_MODES = [
        "encoder",
        "decoder",
        "encoder-decoder",
    ]

    def __init__(
        self,
        dino: str = "dinov2_vits14_ld",
        mode: str = "encoder",
        device: str = "cuda",
    ):
        if dino not in self.SUPPORTED_DINOS:
            raise ValueError(f"`dino` must be one of {self.SUPPORTED_DINOS}")
        if mode not in self.CALL_MODES:
            raise ValueError(f"`mode` must be one of {self.CALL_MODES}")
        self.dino = pt.hub.load("facebookresearch/dinov2:main", dino)
        self.dino.eval()
        self.to(device)
        self.mode = mode
        self.device = device

    def to(self, device: str):
        self.dino.to(device)

    def encode(self, x: pt.Tensor) -> pt.Tensor:
        with pt.inference_mode():
            x = self.dino.backbone(x)
        return x

    def decode(self, x: pt.Tensor) -> pt.Tensor:
        with pt.inference_mode():
            x = self.dino.decode_head.forward_test(x, None)
            x = pt.clamp(
                x,
                min=self.dino.decode_head.min_depth,
                max=self.dino.decode_head.max_depth,
            )
        return x

    def __call__(self, x: pt.Tensor) -> pt.Tensor:
        if self.mode == "encoder":
            x = self.encode(x)
        elif self.mode == "decoder":
            x = self.decode(x)
        elif self.mode == "encoder-decoder":
            x = self.encode(x)
            x = self.decode(x)
        else:
            raise ValueError(f"`mode` must be one of {self.CALL_MODES}")
        return x
