"""
Dataset for generating a video from a single input image + RealEstate10K camera trajectory.

The dataset wraps a user-provided image and a RealEstate10K-format .txt trajectory file
so that they can be fed into the existing DFoT video-generation pipeline.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from PIL import Image

from .base_video import SPLIT


class SingleImageTrajectoryDataset(torch.utils.data.Dataset):
    """
    A minimal dataset that wraps a single image and a camera trajectory .txt file.

    Trajectory format (RealEstate10K):
        Line 1  : YouTube URL (ignored — user supplies their own image)
        Lines 2+: <timestamp> <v0> <v1> ... <v17>
                  18 floats per line:
                    [0:4]  – intrinsics fx, fy, px, py (normalised)
                    [4:6]  – two values skipped (internal RE10K artefact)
                    [6:18] – 3×4 world-to-camera extrinsic matrix, row-major

    After processing, each frame produces a 16-dim vector:
        4 intrinsics + 12 extrinsics  (matching dataset.external_cond_dim=16)

    The dataset always returns the same sample (the input image + sampled trajectory),
    repeated `cfg.num_samples` times so that the validation loop can generate
    multiple videos from one call if desired.
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "validation"):
        self.cfg = cfg
        self.n_frames: int = cfg.n_frames
        self.resolution: int = cfg.resolution
        self.context_length: int = cfg.context_length
        self.num_samples: int = cfg.get("num_samples", 1)

        self.image = self._load_image(Path(cfg.image_path))
        self.poses = self._parse_trajectory(Path(cfg.trajectory_path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load, centre-crop to square, resize. Returns (3, H, W) in [0, 1]."""
        img = Image.open(path).convert("RGB")
        w, h = img.size
        min_dim = min(w, h)
        img = img.crop((
            (w - min_dim) // 2,
            (h - min_dim) // 2,
            (w + min_dim) // 2,
            (h + min_dim) // 2,
        ))
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        return TF.to_tensor(img)  # (3, H, W)

    def _parse_trajectory(self, path: Path) -> torch.Tensor:
        """
        Parse the trajectory file and return (n_frames, 16) processed poses.

        Handles both 18-value raw RE10K files and 16-value pre-processed files.
        """
        cameras = []
        with open(path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if i == 0:
                continue  # first line is YouTube URL
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cam = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            cameras.append(cam)

        if len(cameras) == 0:
            raise ValueError(f"No camera frames found in {path}")

        cameras = np.stack(cameras)                             # (N, raw_dim)
        cameras = torch.tensor(cameras, dtype=torch.float32)
        N, raw_dim = cameras.shape

        # Uniformly subsample / interpolate to exactly n_frames poses
        if N >= self.n_frames:
            indices = torch.linspace(0, N - 1, self.n_frames).round().long()
            cameras = cameras[indices]
        else:
            # Linear interpolation when the trajectory is shorter than n_frames
            t = torch.linspace(0, N - 1, self.n_frames)
            lo = t.floor().long().clamp(0, N - 2)
            hi = (lo + 1).clamp(0, N - 1)
            alpha = (t - lo.float()).unsqueeze(-1)
            cameras = cameras[lo] * (1 - alpha) + cameras[hi] * alpha

        # Mirror _process_external_cond() in datasets/video/realestate10k.py:
        #   keep intrinsics (cols 0-3) and extrinsics (cols 6-17), skip cols 4-5
        if raw_dim == 18:
            poses = torch.cat([cameras[:, :4], cameras[:, 6:]], dim=-1)  # (T, 16)
        elif raw_dim == 16:
            poses = cameras          # already in processed format
        else:
            raise ValueError(
                f"Expected 18 (raw RE10K) or 16 (pre-processed) camera values per line, "
                f"got {raw_dim} in {path}."
            )

        return poses  # (n_frames, 16)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        T = self.n_frames
        # Fill every frame with the input image; the model replaces non-context frames.
        video = self.image.unsqueeze(0).expand(T, -1, -1, -1).clone()  # (T, 3, H, W)
        return {
            "videos": video,                                   # (T, 3, H, W) in [0, 1]
            "conds": self.poses,                               # (T, 16)
            "nonterminal": torch.ones(T, dtype=torch.bool),   # all frames are valid
        }
