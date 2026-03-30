"""
Hybrid Shadow Removal Pipeline
Stage 1 (Classical IP) -> Stage 2 (ShadowFormer) -> Stage 3 (Post-processing)

Results are auto-saved per image in: hybrid_pipeline/results/<image_stem>/

Usage:
    python pipeline.py <input_image> <weights_path>

Example:
    python pipeline.py ../test_images/shadow.jpg \
        ../deep_learning/ShadowFormer/checkpoints/ISTD.pth
"""

import sys
import os
import argparse
import time
import cv2
import numpy as np

# ── Add sibling directories to path ──────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "classical_IP"))
sys.path.insert(0, os.path.join(BASE, "deep_learning"))

from main import detect_shadow_mask      # Stage 1
from remove import run_shadowformer      # Stage 2


# ── Stage 3: Post-processing ──────────────────────────────────────────────────

def post_process(sf_image_path: str, mask_path: str) -> np.ndarray:
    """
    Stage 3: Gentle gamma brightening of residual shadow in ShadowFormer output.
    Works ONLY on the SF output — never blends back the original (which had shadow).
    Gamma < 1 = brightens. Applied softly in shadow region via feathered mask.
    Result is always >= Stage 3 brightness in shadow areas.
    """
    sf   = cv2.imread(sf_image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask.shape[:2] != sf.shape[:2]:
        mask = cv2.resize(mask, (sf.shape[1], sf.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Work in LAB space — only L channel needs brightening
    lab = cv2.cvtColor(sf, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Gamma correction: gamma < 1 makes image brighter
    gamma = 0.82
    l_f = l.astype(np.float32) / 255.0
    l_gamma = np.power(l_f, gamma) * 255.0
    l_gamma = np.clip(l_gamma, 0, 255).astype(np.uint8)

    # Soft feathered mask — apply gamma only INSIDE shadow region
    feather = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0) / 255.0
    l_final = np.clip(
        l.astype(np.float32) * (1.0 - feather) + l_gamma.astype(np.float32) * feather,
        0, 255
    ).astype(np.uint8)

    lab_final = cv2.merge((l_final, a, b))
    return cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)


# ── Comparison Grid ───────────────────────────────────────────────────────────

def make_comparison_grid(original_path, classical_path, sf_path, final_img, output_path):
    """Creates a 4-panel side-by-side comparison image."""
    def load_resize(path, size):
        img = cv2.imread(path)
        return cv2.resize(img, size)

    target_h = 480
    orig = cv2.imread(original_path)
    h, w = orig.shape[:2]
    target_w = int(w * target_h / h)
    sz = (target_w, target_h)

    orig_r   = cv2.resize(orig, sz)
    class_r  = load_resize(classical_path, sz)
    sf_r     = load_resize(sf_path, sz)
    final_r  = cv2.resize(final_img, sz)

    def add_label(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (target_w, 32), (20, 20, 20), -1)
        cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    panels = [
        add_label(orig_r,   "1. Original"),
        add_label(class_r,  "2. Classical IP (Stage 1)"),
        add_label(sf_r,     "3. ShadowFormer (Stage 2)"),
        add_label(final_r,  "4. Final + Post-proc (Stage 3)"),
    ]
    grid = np.hstack(panels)
    cv2.imwrite(output_path, grid)
    print(f"[Pipeline] Comparison grid  -> {output_path}")
    return grid


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_shadow_improvement(orig_path, sf_path, mask_path):
    """Quick brightness delta in the shadow region."""
    orig = cv2.imread(orig_path)
    sf   = cv2.imread(sf_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if orig is None or sf is None or mask is None:
        return None
    shadow = mask > 127
    if not shadow.any():
        return None
    m3 = np.stack([shadow] * 3, axis=-1)
    orig_b = float(orig[m3].mean())
    sf_b   = float(sf[m3].mean())
    shadow_pct = 100.0 * shadow.sum() / shadow.size
    return orig_b, sf_b, shadow_pct


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(image_path: str, weights_path: str, base_results_dir: str):
    # ── Per-image output folder: results/<stem>/ ──────────────────────────────
    stem = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(base_results_dir, stem)
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()

    print("\n" + "="*54)
    print(f"  Hybrid Shadow Removal Pipeline")
    print(f"  Image   : {os.path.basename(image_path)}")
    print(f"  Results : {output_dir}")
    print("="*54)

    # ── Stage 1: Classical shadow mask + brightened baseline ──────────────
    print("\n[Stage 1] Classical IP -- Shadow Detection")
    t1 = time.time()
    original, mask, classical_result, mask_path = detect_shadow_mask(
        image_path, output_dir
    )
    print(f"[Stage 1] Done in {time.time()-t1:.1f}s")
    classical_path = os.path.join(output_dir, "classical_brightened.jpg")

    # ── Stage 2: ShadowFormer deep learning removal ───────────────────────
    print("\n[Stage 2] ShadowFormer -- Deep Learning Removal")
    t2 = time.time()
    sf_output_path = os.path.join(output_dir, "shadowformer_output.jpg")
    run_shadowformer(image_path, mask_path, weights_path, sf_output_path)
    print(f"[Stage 2] Done in {time.time()-t2:.1f}s")

    # ── Stage 3: Post-processing ──────────────────────────────────────────
    print("\n[Stage 3] Post-processing -- Gamma Boost in Shadow Region")
    t3 = time.time()
    final_img = post_process(sf_output_path, mask_path)
    final_path = os.path.join(output_dir, "final_output.jpg")
    cv2.imwrite(final_path, final_img)
    print(f"[Stage 3] Final output      -> {final_path}")
    print(f"[Stage 3] Done in {time.time()-t3:.1f}s")

    # ── Comparison grid ───────────────────────────────────────────────────
    grid_path = os.path.join(output_dir, "comparison_grid.jpg")
    grid = make_comparison_grid(
        image_path, classical_path, sf_output_path, final_img, grid_path
    )

    # ── Quick metrics ─────────────────────────────────────────────────────
    metrics = compute_shadow_improvement(image_path, sf_output_path, mask_path)

    total = time.time() - t_start
    print("\n" + "="*54)
    print("  Pipeline Complete!")
    print("="*54)
    print(f"  Image         : {os.path.basename(image_path)}")
    print(f"  Results folder: {output_dir}")
    if metrics:
        orig_b, sf_b, shadow_pct = metrics
        delta = sf_b - orig_b
        print(f"  Shadow area   : {shadow_pct:.1f}% of image")
        print(f"  Shadow brightness: {orig_b:.1f} -> {sf_b:.1f}  (delta +{delta:.1f})")
    print(f"  Total time    : {total:.1f}s")
    print("="*54)
    print("\n  Files saved:")
    for f in ["shadow_mask.png", "classical_brightened.jpg",
              "shadowformer_output.jpg", "final_output.jpg", "comparison_grid.jpg"]:
        fpath = os.path.join(output_dir, f)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) // 1024
            print(f"    {f:<30} {size_kb:>5} KB")
    print()

    # Show results
    cv2.imshow(f"Result: {stem} (press any key)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Shadow Removal Pipeline")
    parser.add_argument("image",   help="Input shadow image path")
    parser.add_argument("weights", help="Path to ShadowFormer .pth weights")
    parser.add_argument(
        "--results_dir", "-o",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Base results directory (default: hybrid_pipeline/results/)"
    )
    args = parser.parse_args()

    run_pipeline(args.image, args.weights, args.results_dir)
