"""
Evaluation Script — PSNR, SSIM, MAE
Compares shadow removal outputs against ground truth (shadow-free images).

Usage:
    # Compare two single images:
    python evaluate.py --input result.jpg --gt ground_truth.jpg

    # Compare entire folders:
    python evaluate.py --input ./results/ --gt ./ground_truth/

    # Evaluate only shadow region (needs mask):
    python evaluate.py --input result.jpg --gt gt.jpg --mask shadow_mask.png
"""

import os
import sys
import argparse
import cv2
import numpy as np
from glob import glob


# ── Metric Functions ──────────────────────────────────────────────────────────

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio. Higher = better. Unit: dB."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Structural Similarity Index. Range [0,1]. Higher = better."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)
    mu1, mu2 = cv2.blur(i1, (11, 11)), cv2.blur(i2, (11, 11))
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sig1_sq  = cv2.blur(i1 ** 2, (11, 11)) - mu1_sq
    sig2_sq  = cv2.blur(i2 ** 2, (11, 11)) - mu2_sq
    sig12    = cv2.blur(i1 * i2, (11, 11)) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sig12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2))
    return float(np.mean(ssim_map))


def compute_mae(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None) -> float:
    """Mean Absolute Error. Lower = better.
    If mask provided, computes only over shadow region."""
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    if mask is not None:
        m = (mask > 127).astype(bool)
        if m.ndim == 2:
            m = np.stack([m] * 3, axis=-1)
        diff = diff[m]
    return float(np.mean(diff))


# ── Single image evaluation ───────────────────────────────────────────────────

def evaluate_pair(result_path: str, gt_path: str, mask_path: str = None):
    result = cv2.imread(result_path)
    gt     = cv2.imread(gt_path)

    if result is None or gt is None:
        print(f"  [ERROR] Could not load: {result_path} or {gt_path}")
        return None

    # Resize gt to match result if needed
    if result.shape != gt.shape:
        gt = cv2.resize(gt, (result.shape[1], result.shape[0]))

    mask = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape != result.shape[:2]:
            mask = cv2.resize(mask, (result.shape[1], result.shape[0]))

    psnr = compute_psnr(result, gt)
    ssim = compute_ssim(result, gt)
    mae  = compute_mae(result, gt, mask)

    return {"psnr": psnr, "ssim": ssim, "mae": mae}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate shadow removal quality")
    parser.add_argument("--input",  required=True, help="Result image or folder")
    parser.add_argument("--gt",     required=True, help="Ground truth image or folder")
    parser.add_argument("--mask",   default=None,  help="Shadow mask image (optional)")
    args = parser.parse_args()

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    # Single image mode
    if os.path.isfile(args.input):
        metrics = evaluate_pair(args.input, args.gt, args.mask)
        if metrics:
            print(f"\n{'─'*40}")
            print(f"  Image : {os.path.basename(args.input)}")
            print(f"  PSNR  : {metrics['psnr']:.2f} dB  (higher = better)")
            print(f"  SSIM  : {metrics['ssim']:.4f}     (closer to 1 = better)")
            print(f"  MAE   : {metrics['mae']:.2f}       (lower = better)")
            print(f"{'─'*40}\n")
        return

    # Folder mode
    result_files = sorted([
        f for f in glob(os.path.join(args.input, "*"))
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])
    gt_files = sorted([
        f for f in glob(os.path.join(args.gt, "*"))
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])

    if len(result_files) != len(gt_files):
        print(f"[WARNING] File count mismatch: {len(result_files)} results vs {len(gt_files)} GT")

    all_psnr, all_ssim, all_mae = [], [], []
    print(f"\n{'─'*60}")
    print(f"  {'File':<30} {'PSNR':>8} {'SSIM':>8} {'MAE':>8}")
    print(f"{'─'*60}")

    for r, g in zip(result_files, gt_files):
        m = evaluate_pair(r, g, args.mask)
        if m:
            all_psnr.append(m["psnr"])
            all_ssim.append(m["ssim"])
            all_mae.append(m["mae"])
            name = os.path.basename(r)[:30]
            print(f"  {name:<30} {m['psnr']:>8.2f} {m['ssim']:>8.4f} {m['mae']:>8.2f}")

    if all_psnr:
        print(f"{'─'*60}")
        print(f"  {'AVERAGE':<30} {np.mean(all_psnr):>8.2f} {np.mean(all_ssim):>8.4f} {np.mean(all_mae):>8.2f}")
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
