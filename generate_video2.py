#!/usr/bin/env python
"""
generate_video.py — Generate a novel-view synthesis video from a single image
                    and a RealEstate10K camera trajectory.

Modified with stabilized guidance to prevent "rainbow" latent drift artifacts.
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

    # Prediction (Auto-regressive) Guidance - The primary cause of rainbowing
    p.add_argument("--guidance_scale", type=float, default=4.0,
                   help="History guidance scale for prediction (default: 4.0)")
    p.add_argument("--guidance_type", type=str, default="stabilized_vanilla",
                   choices=["vanilla", "stabilized_vanilla", "temporal"],
                   help="History guidance type for prediction (default: stabilized_vanilla)")
    p.add_argument("--stabilization_level", type=float, default=0.02,
                   help="Stabilization level for prediction (prevents rainbow drift) (default: 0.02)")

    # Interpolation Guidance - Lower scale usually keeps the transition cleaner
    p.add_argument("--interp_guidance_scale", type=float, default=1.5,
                   help="History guidance scale for interpolation (default: 1.5)")
    p.add_argument("--interp_guidance_type", type=str, default="vanilla",
                   choices=["vanilla", "stabilized_vanilla", "temporal"],
                   help="History guidance type for interpolation (default: vanilla)")

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
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")

    cameras = []
    timestamps: list[float] = []

    with open(txt_path, "r") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Trajectory file is empty: {txt_path}")

    start_line = 0
    first = lines[0].strip()
    if first and not first[0].isdigit():
        start_line = 1

    for line in lines[start_line:]:
        line = line.strip()
        if not line: continue
        parts = line.split()
        timestamps.append(float(parts[0]))
        cam = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        cameras.append(cam)

    if not cameras:
        raise ValueError(f"No camera frames found in {txt_path}")

    cameras_np = np.stack(cameras)
    timestamps_np = np.array(timestamps, dtype=np.float64)

    t_min, t_max = timestamps_np.min(), timestamps_np.max()
    t_norm = (timestamps_np - t_min) / (t_max - t_min + 1e-12)
    target_t = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)

    raw_dim = cameras_np.shape[1]
    cameras_interp = np.stack(
        [np.interp(target_t, t_norm, cameras_np[:, d]).astype(np.float32)
         for d in range(raw_dim)], axis=-1
    )

    cameras_t = torch.from_numpy(cameras_interp)
    if raw_dim == 18:
        poses = torch.cat([cameras_t[:, :4], cameras_t[:, 6:]], dim=-1)
    else:
        poses = cameras_t

    return poses


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_path: Path, resolution: int = 256) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    min_dim = min(w, h)
    img = img.crop(((w-min_dim)//2, (h-min_dim)//2, (w+min_dim)//2, (h+min_dim)//2))
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return TF.to_tensor(img)


def build_algo_config(args):
    """Compose Hydra config with stabilized guidance parameters."""
    GlobalHydra.instance().clear()
    config_dir = str(PROJECT_ROOT / "configurations")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+name=generate_video",
                "wandb.entity=dummy", "wandb.mode=disabled",
                "dataset=realestate10k",
                "algorithm=dfot_video_pose",
                "experiment=video_generation",
                "++algorithm.diffusion.is_continuous=true",
                "++algorithm.backbone.use_fourier_noise_embedding=true",
                "++algorithm.diffusion.precond_scale=0.125",
                # Model Architecture matching RE10K weights
                "++algorithm.backbone.channels=[128,256,576,1152]",
                "++algorithm.backbone.num_updown_blocks=[3,3,6]",
                "++algorithm.backbone.num_mid_blocks=20",
                "++algorithm.backbone.num_heads=9",
                f"dataset.n_frames={args.n_frames}",
                "dataset.context_length=1",
                "dataset.frame_skip=1",
                # Prediction Task Guidance (The fix for rainbows)
                f"++algorithm.tasks.prediction.history_guidance.name={args.guidance_type}",
                f"++algorithm.tasks.prediction.history_guidance.guidance_scale={args.guidance_scale}",
                f"++algorithm.tasks.prediction.history_guidance.stabilization_level={args.stabilization_level}",
                # Interpolation Task Guidance
                f"++algorithm.tasks.interpolation.history_guidance.name={args.interp_guidance_type}",
                f"++algorithm.tasks.interpolation.history_guidance.guidance_scale={args.interp_guidance_scale}",
            ],
        )
    return cfg.algorithm


def load_model_weights(model: DFoTVideoPose, checkpoint_str: str, device: str) -> None:
    if checkpoint_str.startswith("pretrained:"):
        ckpt_path = download_pretrained(checkpoint_str)
    else:
        ckpt_path = Path(checkpoint_str)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    is_training_ckpt = (not ckpt.get("pretrained_ema", False) and
                        bool(ckpt.get("optimizer_states")) and
                        "ema" in ckpt["optimizer_states"][0])
    if is_training_ckpt:
        model.should_validate_ema_weights = True

    model.on_load_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"], strict=False)


@torch.no_grad()
def generate(model, image, poses, device):
    T = poses.shape[0]
    n_ctx = model.n_context_frames

    video = image.unsqueeze(0).expand(T, -1, -1, -1).clone()
    xs = video.unsqueeze(0).to(device)
    conditions = poses.unsqueeze(0).to(device)

    xs = model._normalize_x(xs)

    # Sliding window logic
    max_tokens = model.max_tokens
    sliding_ctx = model.cfg.tasks.prediction.sliding_context_len or max_tokens // 2
    stride = max_tokens - sliding_ctx

    if T > max_tokens and (T - max_tokens) % stride != 0:
        T_safe = T + stride - (T - max_tokens) % stride
        n_pad = T_safe - T
        xs = torch.cat([xs, xs[:, -1:].expand(-1, n_pad, *[-1]*(xs.dim()-2))], dim=1)
        conditions = torch.cat([conditions, conditions[:, -1:].expand(-1, n_pad, -1)], dim=1)

    xs_pred = model._predict_videos(xs, conditions=conditions)
    xs_pred = xs_pred[:, :T]
    xs_pred = model._unnormalize_x(xs_pred)

    ctx = image.to(device).unsqueeze(0).unsqueeze(0)
    xs_pred[:, :n_ctx] = ctx.expand(1, n_ctx, -1, -1, -1)

    frames = (xs_pred[0].permute(0, 2, 3, 1) * 255).clamp(0, 255).byte().cpu()
    return frames


def _write_video(path, frames, fps):
    import subprocess
    H, W = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "pipe:0", "-vcodec", "libx264", "-pix_fmt", "yuv420p", path
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for frame in frames.numpy():
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        proc.wait()
    except Exception as e:
        print(f"Video write failed: {e}")


def main():
    args = parse_args()

    print("[1/5] Building configuration ...")
    algo_cfg = build_algo_config(args)

    x_shape = OmegaConf.select(algo_cfg, "x_shape", default=None)
    resolution = int(x_shape[1]) if x_shape and len(x_shape) == 3 else 256

    print("[2/5] Building model ...")
    model = DFoTVideoPose(algo_cfg).to(args.device)
    model.eval()

    print("[3/5] Loading checkpoint ...")
    load_model_weights(model, args.checkpoint, args.device)

    print("[4/5] Loading inputs ...")
    image = load_image(args.image, resolution)
    poses = parse_trajectory(args.trajectory, args.n_frames)

    print(f"[5/5] Generating {args.n_frames} frames on {args.device} ...")
    frames = generate(model, image, poses, args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_video(str(args.output), frames, fps=args.fps)
    print(f"\nDone — video saved to: {args.output}")

if __name__ == "__main__":
    main()
