import torch
import torch.nn as nn
from typing import Dict, Union, Optional
from numbers import Number

try:
    from moge.model.v2 import MoGeModel
except ImportError:
    MoGeModel = None

from vipe.utils.misc import unpack_optional
from vipe.utils.cameras import CameraType
from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType

def focal_length_to_fov_degrees(focal_length: float, image_width: float) -> float:
    """Compute horizontal field of view from focal length."""
    fov_rad = 2 * torch.atan(torch.tensor(image_width / (2 * focal_length)))
    fov_deg = torch.rad2deg(fov_rad)
    return fov_deg.item()

class MoGeV2Model(DepthEstimationModel, nn.Module):
    def __init__(self, pretrained: str = "Ruicheng/moge-2-vitl"):
        super().__init__()
        nn.Module.__init__(self)
        if MoGeModel is None:
            raise ImportError("MoGe not found. Ensure vipe/thirdparty/moge is in sys.path")
        self.model = MoGeModel.from_pretrained(pretrained)
        self.model.eval()
        self.model.to("cuda")

    @property
    def depth_type(self) -> DepthType:
        return DepthType.MODEL_METRIC_DEPTH

    def forward(
        self, 
        image: torch.Tensor, 
        fov_x: Optional[float] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        image: (B, 3, H, W)
        fov_x: horizontal fov in degrees (optional)
        """
        dtype = torch.float16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32
        
        # If input has 3 dims, unsqueeze
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        with torch.amp.autocast('cuda', dtype=dtype):
             return self.model.infer(image, fov_x=fov_x, use_fp16=(dtype==torch.float16), **kwargs)

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb: torch.Tensor = unpack_optional(src.rgb)
        assert rgb.dtype == torch.float32, "Input image should be float32"
        assert src.camera_type == CameraType.PINHOLE, "MoGe only supports pinhole cameras"

        focal_length: float = unpack_optional(src.intrinsics)[0].item()

        if rgb.dim() == 3:
            rgb, batch_dim = rgb[None], False
        else:
            batch_dim = True

        w = rgb.shape[2]
        
        # MoGe expects (B, 3, H, W). src.rgb is usually (3, H, W) or (B, 3, H, W).
        # But wait, in vipe pipeline, frames are usually (H, W, 3) numpy or tensor?
        # Let's check `vipe/streams/base.py` or how `rgb` is passed.
        # In `moge.py` (v1), it did `input_image_for_depth = rgb.moveaxis(-1, 1)`.
        # This suggests `rgb` coming in is (..., H, W, 3)?
        # Let's check `DepthEstimationInput` definition again.
        # `rgb: The source image ([B,], H, W, 3)`
        
        # So we need to permute.
        if rgb.shape[-1] == 3:
             input_image = rgb.permute(0, 3, 1, 2) # (B, 3, H, W)
        else:
             input_image = rgb # Assume already correct if not channel last

        fov_deg = focal_length_to_fov_degrees(focal_length, w)

        res = self.forward(input_image, fov_x=fov_deg)
        
        # MoGe2 returns 'depth' key which is metric depth
        # depth shape: (B, H, W)
        moge_depth = res['depth']
        moge_mask = res.get('mask', None) # (B, H, W) or similar
        
        # Process depth
        moge_depth = torch.nan_to_num(moge_depth, nan=1e4)
        moge_depth = torch.clamp(moge_depth, min=0, max=1e4)

        if moge_mask is not None:
             if moge_mask.dim() == 4: # (B, 1, H, W)
                 moge_mask = moge_mask.squeeze(1)
             moge_depth = moge_depth * moge_mask.float()

        if not batch_dim:
            moge_depth = moge_depth.squeeze(0)
            
        return DepthEstimationResult(metric_depth=moge_depth)
