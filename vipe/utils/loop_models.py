import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms as T
import math
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ==========================
# SALAD Aggregator
# ==========================

def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    M = M / reg
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()
    return M + u.unsqueeze(2) + v.unsqueeze(1)

def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    batch_size, m, n = S.size()
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a = norm.expand(m + 1).contiguous()
    log_a = log_a.clone()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), norm.expand(n).contiguous().expand(batch_size, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm

class SALAD(nn.Module):
    def __init__(self, num_channels=1536, num_clusters=64, cluster_dim=128, token_dim=256, dropout=0.3):
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        x, t = x
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)[:, :-1, :]
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        f = torch.cat([
            nn.functional.normalize(t, p=2, dim=-1),
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)
        return nn.functional.normalize(f, p=2, dim=-1)

# ==========================
# Backbone (DINOv2)
# ==========================

class DINOv2Backbone(nn.Module):
    def __init__(self, arch='dinov2_vitb14'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', arch)
        self.num_channels = self.model.embed_dim * 2 # Concatenation of token and features? No, check logic.
        # vggt-long uses return_token=True.
        
    def forward(self, x):
        # DINOv2 forward_features returns dict
        out = self.model.forward_features(x)
        # x_norm_clstoken, x_norm_patchtokens
        t = out['x_norm_clstoken']
        f = out['x_norm_patchtokens']
        # Reshape f to (B, C, H, W)
        B, N, C = f.shape
        H = W = int(math.sqrt(N))
        f = f.permute(0, 2, 1).reshape(B, C, H, W)
        return f, t

# ==========================
# VPR Model
# ==========================

class VPRModel(pl.LightningModule):
    def __init__(self, backbone_arch='dinov2_vitb14', agg_config=None):
        super().__init__()
        if agg_config is None:
            agg_config = {'num_channels': 768, 'num_clusters': 64, 'cluster_dim': 128, 'token_dim': 256}
            
        self.backbone = DINOv2Backbone(backbone_arch)
        self.aggregator = SALAD(**agg_config)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

# ==========================
# Loop Detector Wrapper
# ==========================

class SaladLoopDetector:
    def __init__(self, ckpt_path, image_size=[336, 336], batch_size=32, 
                 similarity_threshold=0.85, top_k=5, use_nms=True, nms_threshold=25):
        self.ckpt_path = ckpt_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.use_nms = use_nms
        self.nms_threshold = nms_threshold
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = self._input_transform(image_size)
        self.descriptors = None
        self.loop_closures = []

    def _input_transform(self, image_size):
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        if self.model is None:
            # We need to construct the model structure matching the checkpoint
            # Assuming checkpoint is from VGGT-Long which uses specific structure.
            # If we cannot load weights due to structure mismatch, we might need to be careful.
            # The vggt-long VPRModel uses specific keys.
            
            # Let's try to load state dict and adapt if needed.
            # Or assume user provides correct path.
            
            try:
                self.model = VPRModel()
                # Load weights
                # vggt checkpoint might be a lightning checkpoint or state dict.
                checkpoint = torch.load(self.ckpt_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                    
                # Fix keys if necessary (e.g. remove "model." prefix if any)
                # vggt uses "backbone." and "aggregator." prefixes in VPRModel.
                # My implementation uses same names.
                self.model.load_state_dict(state_dict, strict=False) # strict=False to allow minor diffs in backbone internal names
                self.model.to(self.device).eval()
                logger.info(f"Loaded SALAD model from {self.ckpt_path}")
            except Exception as e:
                logger.error(f"Failed to load SALAD model: {e}")
                raise

    def extract_features(self, video_stream):
        self.load_model()
        descriptors = []
        
        # Iterate over stream
        # video_stream might not support random access efficiently for batching if it's RawMp4Stream.
        # But we implemented __getitem__.
        # We can also just iterate.
        
        batch_imgs = []
        
        # Using simple iteration for robustness
        total = len(video_stream)
        for i, frame in enumerate(video_stream):
            img = Image.fromarray((frame.rgb.cpu().numpy() * 255).astype('uint8'))
            img_tensor = self.transform(img)
            batch_imgs.append(img_tensor)
            
            if len(batch_imgs) == self.batch_size or i == total - 1:
                batch = torch.stack(batch_imgs).to(self.device)
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        desc = self.model(batch)
                descriptors.append(desc.cpu())
                batch_imgs = []
                
        if descriptors:
            self.descriptors = torch.cat(descriptors)
        return self.descriptors

    def find_loops(self):
        if self.descriptors is None:
            return []
            
        # Use FAISS or Brute Force torch
        # Brute force is fine for < 10k frames
        feats = self.descriptors
        sim = feats @ feats.T # (N, N)
        
        N = len(feats)
        mask = torch.ones((N, N), dtype=torch.bool, device=sim.device)
        exclude_window = 10 # Hardcoded exclusion from vggt-long
        for i in range(N):
            s = max(0, i - exclude_window)
            e = min(N, i + exclude_window)
            mask[i, s:e] = False
        mask = torch.triu(mask)
        
        candidates = torch.nonzero((sim > self.similarity_threshold) & mask)
        cand_scores = sim[candidates[:, 0], candidates[:, 1]]
        
        # Sort desc
        sorted_idx = torch.argsort(cand_scores, descending=True)
        candidates = candidates[sorted_idx]
        cand_scores = cand_scores[sorted_idx]
        
        # NMS
        selected = []
        suppressed = set()
        
        candidates_list = []
        for idx in range(len(candidates)):
            i, j = candidates[idx].tolist()
            candidates_list.append((i, j, cand_scores[idx].item()))
            
        # Apply vggt NMS logic exactly
        # Note: vggt NMS logic in LoopModel.py
        
        for idx1, idx2, score in candidates_list:
            if idx1 in suppressed or idx2 in suppressed:
                continue
                
            selected.append((idx1, idx2, score))
            
            # Suppress ranges
            start1 = max(0, idx1 - self.nms_threshold)
            end1 = min(idx1 + self.nms_threshold + 1, idx2) # vggt logic
            for k in range(start1, end1): suppressed.add(k)
            
            # vggt: end2 = min(idx2 + nms_threshold + 1, max_frame + 1)
            # max_frame here is N-1
            start2 = max(idx1 + 1, idx2 - self.nms_threshold)
            end2 = min(idx2 + self.nms_threshold + 1, N)
            for k in range(start2, end2): suppressed.add(k)
            
        self.loop_closures = [(i, j) for i, j, _ in selected]
        return self.loop_closures


