#!/usr/bin/env python
"""
generate_video.py — Generate a novel-view synthesis video from a single image
                    and a RealEstate10K camera trajectory.

Run from the project root:

    python generate_video.py \\
        --image   /path/to/frame.jpg \\
        --trajectory /path/to/clip.txt \\
        --output  output.mp4 \\
        [--n_frames 50] \\
        [--checkpoint pretrained:DFoT_RE10K.ckpt] \\
        [--guidance_scale 4.0] \\
        [--guidance_type stabilized_vanilla] \\
        [--fps 10] \\
        [--device cuda]

Trajectory file format (RealEstate10K):
    Line 1  : YouTube URL  (ignored — we use --image instead)
    Lines 2+: <timestamp_us> <fx> <fy> <px> <py> <v4> <v5> \\
              <r11> <r12> <r13> <tx> <r21> <r22> <r23> <ty> \\
              <r31> <r32> <r33> <tz>

    • Intrinsics fx, fy, px, py are normalised: left-top=(0,0), right-bottom=(1,1)
    • [R|t] is a 3×4 world-to-camera matrix stored row-major
    • v4, v5 are two values present in the raw RE10K format that are ignored
      (same as the dataset loader in datasets/video/realestate10k.py)
    • 16-value pre-processed files (without v4/v5) are also accepted
"""

import argparse
import sys
import os
from pathlib import Path

# ── Ensure imports resolve from the project root ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.io import write_video
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from algorithms.dfot import DFoTVideoPose
from utils.ckpt_utils import download_pretrained


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a novel-view video from one image + RE10K trajectory",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--image", type=Path, required=True,
                   help="Starting image (from RE10K test set or any compatible image)")
    p.add_argument("--trajectory", type=Path, required=True,
                   help="RealEstate10K .txt trajectory file")
    p.add_argument("--output", type=Path, default=Path("output.mp4"),
                   help="Output video path  (default: output.mp4)")
    p.add_argument("--n_frames", type=int, default=50,
                   help="Number of frames to generate  (default: 50)")
    p.add_argument("--checkpoint", type=str, default="pretrained:DFoT_RE10K.ckpt",
                   help='Checkpoint to use.\n'
                        '  "pretrained:DFoT_RE10K.ckpt" – auto-download from HuggingFace\n'
                        '  /path/to/model.ckpt           – local Lightning checkpoint')
    p.add_argument("--guidance_scale", type=float, default=4.0,
                   help="History guidance scale  (default: 4.0)")
    p.add_argument("--guidance_type", type=str, default="stabilized_vanilla",
                   choices=["vanilla", "stabilized_vanilla", "temporal"],
                   help="History guidance type  (default: stabilized_vanilla)")
    p.add_argument("--fps", type=int, default=10,
                   help="Output video frame rate  (default: 10)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Inference device  (default: cuda if available, else cpu)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_trajectory(txt_path: Path, n_frames: int) -> torch.Tensor:
    """
    Parse a RealEstate10K-format .txt file and return (n_frames, 16) camera poses.

    The 16-dim output is:  [fx, fy, px, py, r11, r12, r13, tx,
                                              r21, r22, r23, ty,
                                              r31, r32, r33, tz]
    (4 intrinsics + 12 extrinsics, matching dataset.external_cond_dim=16)

    Accepted raw formats:
      18 values/line – raw RE10K (values at positions 4-5 are silently dropped,
                       mirroring _process_external_cond in realestate10k.py)
      16 values/line – pre-processed (used as-is)
    """
    cameras = []
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i == 0:
            continue          # first line is the YouTube URL
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cam = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        cameras.append(cam)

    if not cameras:
        raise ValueError(f"No camera frames found in {txt_path}")

    cameras = np.stack(cameras)                            # (N, raw_dim)
    cameras = torch.tensor(cameras, dtype=torch.float32)
    N, raw_dim = cameras.shape

    # Subsample or interpolate to exactly n_frames poses
    if N >= n_frames:
        idx = torch.linspace(0, N - 1, n_frames).round().long()
        cameras = cameras[idx]
    else:
        t = torch.linspace(0, N - 1, n_frames)
        lo = t.floor().long().clamp(0, N - 2)
        hi = (lo + 1).clamp(0, N - 1)
        alpha = (t - lo.float()).unsqueeze(-1)
        cameras = cameras[lo] * (1 - alpha) + cameras[hi] * alpha

    # Convert to 16-dim processed format (matches _process_external_cond)
    if raw_dim == 18:
        poses = torch.cat([cameras[:, :4], cameras[:, 6:]], dim=-1)   # (T, 16)
    elif raw_dim == 16:
        poses = cameras
    else:
        raise ValueError(
            f"{txt_path}: expected 18 (raw RE10K) or 16 (pre-processed) camera "
            f"values per line, got {raw_dim}."
        )
    return poses   # (n_frames, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_path: Path, resolution: int = 256) -> torch.Tensor:
    """Load, centre-crop to square, resize. Returns (3, H, W) in [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    min_dim = min(w, h)
    img = img.crop((
        (w - min_dim) // 2, (h - min_dim) // 2,
        (w + min_dim) // 2, (h + min_dim) // 2,
    ))
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return TF.to_tensor(img)   # (3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Config + model loading
# ─────────────────────────────────────────────────────────────────────────────

def build_algo_config(n_frames: int, guidance_type: str, guidance_scale: float):
    """
    Compose the algorithm DictConfig needed by DFoTVideoPose.

    Uses Hydra's programmatic compose API so that all YAML defaults,
    interpolations (${dataset.xxx}), and dataset-experiment overrides
    (e.g. backbone architecture) are resolved identically to the training run.
    """
    GlobalHydra.instance().clear()
    config_dir = str(PROJECT_ROOT / "configurations")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+name=generate_video",
                "wandb.entity=dummy",              # required field; unused at inference
                "wandb.mode=disabled",
                # Use realestate10k so that realestate10k_video_generation.yaml is
                # picked up automatically by the optional dataset_experiment override.
                # That file sets the backbone architecture, diffusion schedule, etc.
                # that must match the pretrained checkpoint.
                # (We only use cfg.algorithm from this compose; no dataset is loaded.)
                "dataset=realestate10k",
                "algorithm=dfot_video_pose",
                "experiment=video_generation",
                # Expand the @diffusion/continuous shortcut inline
                "++algorithm.diffusion.is_continuous=true",
                "++algorithm.backbone.use_fourier_noise_embedding=true",
                "++algorithm.diffusion.precond_scale=0.125",
                # Override the dataset fields that the algo config interpolates
                f"dataset.n_frames={n_frames}",
                "dataset.context_length=1",
                "dataset.frame_skip=1",
                # Inference guidance settings
                f"++algorithm.tasks.prediction.history_guidance.name={guidance_type}",
                f"++algorithm.tasks.prediction.history_guidance.guidance_scale={guidance_scale}",
            ],
        )
    return cfg.algorithm


def load_model_weights(model: DFoTVideoPose, checkpoint_str: str, device: str) -> None:
    """
    Load weights from a Lightning checkpoint into the (already-built) model.

    Handles two checkpoint flavours:
    - Pretrained release  (pretrained_ema=True):  EMA weights are already in state_dict.
    - Training checkpoint (pretrained_ema absent): EMA weights live in optimizer_states[0]['ema']
      and must be extracted first.
    """
    if checkpoint_str.startswith("pretrained:"):
        print(f"  Downloading pretrained checkpoint: {checkpoint_str}")
        ckpt_path = download_pretrained(checkpoint_str)
    else:
        ckpt_path = Path(checkpoint_str)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"  Loading weights from {ckpt_path} ...")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # For training checkpoints, tell the model to swap in EMA weights before loading.
    # (The EMA callback normally does this via Trainer.setup(), which we bypass here.)
    is_training_ckpt = (
        not ckpt.get("pretrained_ema", False)
        and bool(ckpt.get("optimizer_states"))
        and "ema" in ckpt["optimizer_states"][0]
    )
    if is_training_ckpt:
        model.should_validate_ema_weights = True

    # on_load_checkpoint remaps / filters state_dict keys to match the current model
    model.on_load_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"], strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: DFoTVideoPose,
    image: torch.Tensor,     # (3, H, W) in [0, 1]
    poses: torch.Tensor,     # (T, 16)
    device: str,
) -> torch.Tensor:
    """
    Run DFoT prediction to generate a video from one context frame + trajectory.

    Steps mirror what on_after_batch_transfer + _sample_all_videos do inside the
    Lightning validation loop, but without needing a Trainer or WandB logger.

    Returns:
        (T, H, W, 3) uint8 tensor, values in [0, 255]
    """
    T = poses.shape[0]

    # Build the full-length input tensor.
    # The model uses xs[:, :n_context_tokens] as the conditioning context;
    # the remaining tokens will be diffused / generated.
    video = image.unsqueeze(0).expand(T, -1, -1, -1).clone()   # (T, 3, H, W)
    xs = video.unsqueeze(0).to(device)                          # (1, T, 3, H, W)
    conditions = poses.unsqueeze(0).to(device)                  # (1, T, 16)

    # Normalise exactly as on_after_batch_transfer does
    xs = model._normalize_x(xs)

    # Handle latent-diffusion models (RE10K pretrained is pixel-space, but be safe)
    if model.is_latent_diffusion:
        n_ctx = model.n_context_frames
        ctx_latents = model._encode(xs[:, :n_ctx])
        xs_latent = torch.zeros(1, T, *ctx_latents.shape[2:], device=device,
                                dtype=ctx_latents.dtype)
        xs_latent[:, :n_ctx] = ctx_latents
        xs = xs_latent

    # Generate all frames via sliding-window diffusion rollout
    xs_pred = model._predict_videos(xs, conditions=conditions)

    # Decode latents → pixels (no-op for pixel-space models)
    if model.is_latent_diffusion:
        xs_pred = model._decode(xs_pred)

    # Unnormalize to [0, 1]
    xs_pred = model._unnormalize_x(xs_pred)                    # (1, T, 3, H, W)

    # Paste the ground-truth context frame(s) back (same as validation_step does)
    ctx = image.to(device).unsqueeze(0).unsqueeze(0)           # (1, 1, 3, H, W)
    xs_pred[:, :model.n_context_frames] = ctx.expand(
        1, model.n_context_frames, -1, -1, -1
    )

    # Convert to (T, H, W, 3) uint8
    frames = (xs_pred[0].permute(0, 2, 3, 1) * 255).clamp(0, 255).byte().cpu()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Configuration ──────────────────────────────────────────────────────
    print("[1/5] Building configuration ...")
    algo_cfg = build_algo_config(args.n_frames, args.guidance_type, args.guidance_scale)
    resolution = OmegaConf.select(algo_cfg, "x_shape.1", default=256)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("[2/5] Building model ...")
    model = DFoTVideoPose(algo_cfg)
    model = model.to(args.device)
    model.eval()

    # ── 3. Checkpoint ─────────────────────────────────────────────────────────
    print("[3/5] Loading checkpoint ...")
    load_model_weights(model, args.checkpoint, args.device)

    # ── 4. Inputs ─────────────────────────────────────────────────────────────
    print(f"[4/5] Loading inputs ...")
    print(f"      image      : {args.image}")
    print(f"      trajectory : {args.trajectory}  ({args.n_frames} frames)")
    image = load_image(args.image, resolution)
    poses = parse_trajectory(args.trajectory, args.n_frames)

    # ── 5. Generate & save ────────────────────────────────────────────────────
    print(f"[5/5] Generating {args.n_frames} frames on {args.device} ...")
    frames = generate(model, image, poses, args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(args.output), frames, fps=args.fps)
    print(f"\nDone — video saved to: {args.output}")


if __name__ == "__main__":
    main()
