# generate keyframes, then interpolate (instead of sequential frame generation)
#!/usr/bin/env python
"""
generate_video.py — Hierarchical Generation (Keyframes + Interpolation)
to match official DFoT rollout stability.
"""

import argparse
import sys
import os
from pathlib import Path

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
    p = argparse.ArgumentParser(description="Hierarchical DFoT Video Generation")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("output.mp4"))
    p.add_argument("--n_frames", type=int, default=64, help="Model thrives on 100+ frames")
    p.add_argument("--checkpoint", type=str, default="pretrained:DFoT_RE10K.ckpt")

    # Hierarchical Settings
    p.add_argument("--keyframe_density", type=float, default=0.0625,
                   help="Density of keyframes (0.0625 = every 16th frame)")

    # Prediction (Keyframe) Guidance
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--guidance_type", type=str, default="stabilized_vanilla")
    p.add_argument("--stabilization_level", type=float, default=0.02)

    # Interpolation (Filling) Guidance
    p.add_argument("--interp_guidance_scale", type=float, default=1.5)
    p.add_argument("--interp_guidance_type", type=str, default="vanilla")

    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# [Keep parse_trajectory and load_image functions from your original script]
# ─────────────────────────────────────────────────────────────────────────────
# Utils (Abbreviated for brevity)
# ─────────────────────────────────────────────────────────────────────────────

def parse_trajectory(txt_path: Path, n_frames: int) -> torch.Tensor:
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and l.strip()[0].isdigit()]
    cameras = np.stack([np.array([float(x) for x in l.split()[1:]]) for l in lines])
    ts = np.linspace(0, 1, len(cameras))
    target_t = np.linspace(0, 1, n_frames)
    interp = np.stack([np.interp(target_t, ts, cameras[:, d]) for d in range(cameras.shape[1])], axis=-1)
    t = torch.from_numpy(interp).float()
    return torch.cat([t[:, :4], t[:, 6:]], dim=-1) if t.shape[1] == 18 else t

def load_image(image_path: Path, resolution: int = 256) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    min_dim = min(w, h); img = img.crop(((w-min_dim)//2, (h-min_dim)//2, (w+min_dim)//2, (h+min_dim)//2))
    return TF.to_tensor(img.resize((resolution, resolution), Image.LANCZOS))

def build_algo_config(args):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configurations"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "dataset=realestate10k",
            "algorithm=dfot_video_pose",
            "experiment=video_generation",
            "++algorithm.diffusion.is_continuous=true",
            "++algorithm.backbone.use_fourier_noise_embedding=true",
            "++algorithm.diffusion.precond_scale=0.125",
            "++algorithm.backbone.channels=[128,256,576,1152]",
            "++algorithm.backbone.num_updown_blocks=[3,3,6]",
            "++algorithm.backbone.num_mid_blocks=20",
            "++algorithm.backbone.num_heads=9",
            "dataset.n_frames=25",          # ← match what the checkpoint was trained on
            "dataset.context_length=1",
            # lower guidance scales reduce CFG memory overhead
            f"++algorithm.tasks.prediction.history_guidance.name={args.guidance_type}",
            f"++algorithm.tasks.prediction.history_guidance.guidance_scale={args.guidance_scale}",
            f"++algorithm.tasks.prediction.history_guidance.stabilization_level={args.stabilization_level}",
            f"++algorithm.tasks.interpolation.history_guidance.name={args.interp_guidance_type}",
            f"++algorithm.tasks.interpolation.history_guidance.guidance_scale={args.interp_guidance_scale}",
            "++algorithm.diffusion.sampling_timesteps=10",   # ← biggest memory/speed lever
            "++algorithm.tasks.interpolation.max_batch_size=1",  # ← prevents interpolation OOM
        ])
    return cfg.algorithm

def load_model_weights(model, checkpoint_str, device):
    ckpt_path = download_pretrained(checkpoint_str) if checkpoint_str.startswith("pretrained:") else Path(checkpoint_str)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.on_load_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"], strict=False)

# ─────────────────────────────────────────────────────────────────────────────
# The Fix: Hierarchical Generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_hierarchical(model, image, poses, args):
    T = poses.shape[0]   # e.g. 64 — this is fine, sliding window handles it
    device = args.device

    xs = image.unsqueeze(0).expand(T, -1, -1, -1).unsqueeze(0).to(device)
    xs = model._normalize_x(xs)
    conditions = poses.unsqueeze(0).to(device)

    model.cfg.n_frames = T
    original_density = model.cfg.tasks.prediction.keyframe_density
    model.cfg.tasks.prediction.keyframe_density = args.keyframe_density
    original_scl = model.cfg.tasks.prediction.sliding_context_len
    model.cfg.tasks.prediction.sliding_context_len = model.n_context_tokens

    xs_pred = model._predict_videos(xs, conditions=conditions)
    torch.cuda.empty_cache()

    model.cfg.tasks.prediction.keyframe_density = original_density
    model.cfg.tasks.prediction.sliding_context_len = original_scl

    xs_pred = model._unnormalize_x(xs_pred)
    xs_pred[0, 0] = image.to(device)
    frames = (xs_pred[0].permute(0, 2, 3, 1) * 255).clamp(0, 255).byte().cpu()
    return frames

def _write_video(path, frames, fps):
    import subprocess
    H, W = frames.shape[1], frames.shape[2]
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps),
           "-i", "-", "-vcodec", "libx264", "-pix_fmt", "yuv420p", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames.numpy(): proc.stdin.write(f.tobytes())
    proc.stdin.close(); proc.wait()

def main():
    args = parse_args()
    print("[1/5] Building config...")
    algo_cfg = build_algo_config(args)
    print("[2/5] Building model...")
    model = DFoTVideoPose(algo_cfg).to(args.device).eval()
    print("[3/5] Loading weights...")
    load_model_weights(model, args.checkpoint, args.device)
    print("[4/5] Loading inputs...")
    image = load_image(args.image)
    poses = parse_trajectory(args.trajectory, args.n_frames)
    print(f"[5/5] Generating {args.n_frames} frames (Hierarchical)...")
    frames = generate_hierarchical(model, image, poses, args)
    _write_video(str(args.output), frames, args.fps)
    print(f"Done: {args.output}")

if __name__ == "__main__":
    main()
