"""
Stage 2: ShadowFormer — Deep Learning Shadow Removal
Minimal single-image inference wrapper around ShadowFormer (AAAI 2023).

Mirrors the official test.py approach using ShadowFormer's own utils.
Handles: padding to win_size multiples, model loading, inference, unpadding.

Usage:
    python remove.py <input_image> <shadow_mask> <weights.pth> [--output path]

Example:
    python remove.py ../test_images/shadow.jpg shadow_mask.png \
        ShadowFormer/checkpoints/ISTD.pth --output ../results/sf_out.jpg
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# ── Add ShadowFormer source to path ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SF_DIR = os.path.join(SCRIPT_DIR, "ShadowFormer")
sys.path.insert(0, SF_DIR)

import utils                          # ShadowFormer's utils package
from utils.model_utils import load_checkpoint, get_arch


# ── Fake args object to reuse get_arch() ─────────────────────────────────────
class _Args:
    arch            = "ShadowFormer"
    embed_dim       = 32
    win_size        = 10
    token_projection = "linear"
    token_mlp       = "leff"
    train_ps        = 320             # used only for model instantiation


# ── Core inference function ───────────────────────────────────────────────────

def run_shadowformer(
    image_path: str,
    mask_path: str,
    weights_path: str,
    output_path: str = None,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 2] Device        : {device}")
    print(f"[Stage 2] Input image   : {image_path}")
    print(f"[Stage 2] Shadow mask   : {mask_path}")
    print(f"[Stage 2] Weights       : {weights_path}")

    # ── Load & normalise inputs ───────────────────────────────────────────────
    img_bgr  = cv2.imread(image_path)
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    if mask_raw is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")

    # Resize mask to image size if different
    if mask_raw.shape[:2] != img_bgr.shape[:2]:
        mask_raw = cv2.resize(mask_raw, (img_bgr.shape[1], img_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # BGR → RGB, uint8 → float32 [0,1], HWC → CHW → BCHW
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_t    = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)   # [1,3,H,W]

    # Mask: [0,1] float, 1-channel  BCHW
    mask_f   = (mask_raw.astype(np.float32) / 255.0)
    mask_t   = torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0)        # [1,1,H,W]

    height, width = img_t.shape[2], img_t.shape[3]

    # ── Pad to multiple of win_size*8 = 80 ───────────────────────────────────
    img_multiple_of = 8 * _Args.win_size  # 80
    H = ((height + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
    W = ((width  + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
    padh = H - height
    padw = W - width
    if padh > 0 or padw > 0:
        img_t  = F.pad(img_t,  (0, padw, 0, padh), mode="reflect")
        mask_t = F.pad(mask_t, (0, padw, 0, padh), mode="reflect")

    img_t  = img_t.to(device)
    mask_t = mask_t.to(device)

    # ── Load model ────────────────────────────────────────────────────────────
    print("[Stage 2] Loading model ...")
    model = get_arch(_Args)
    model = torch.nn.DataParallel(model)
    load_checkpoint(model, weights_path)
    model.to(device)
    model.eval()
    print("[Stage 2] Model ready. Running inference ...")

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        restored = model(img_t, mask_t)   # returns [1,3,H_pad,W_pad]

    restored = torch.clamp(restored, 0, 1)

    # ── Unpad & convert back ──────────────────────────────────────────────────
    restored = restored[:, :, :height, :width]   # remove padding
    out_np = restored.squeeze(0).cpu().numpy()    # [3,H,W]
    out_np = (out_np.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    # ── Save ──────────────────────────────────────────────────────────────────
    if output_path is None:
        base = os.path.splitext(image_path)[0]
        output_path = base + "_shadowformer.jpg"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, out_bgr)
    print(f"[Stage 2] Output saved  -> {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShadowFormer single-image inference")
    parser.add_argument("image",    help="Input shadow image")
    parser.add_argument("mask",     help="Binary shadow mask PNG")
    parser.add_argument("weights",  help="Pretrained .pth weights path")
    parser.add_argument("--output", "-o", default=None, help="Output image path")
    a = parser.parse_args()
    run_shadowformer(a.image, a.mask, a.weights, a.output)
