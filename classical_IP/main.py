"""
Stage 1: Classical IP — Shadow Detection
Detects shadow mask using LAB color space + Otsu thresholding + morphology.
Outputs: shadow_mask.png (binary) for use by Stage 2 (ShadowFormer).

Usage:
    python main.py <input_image_path> [output_dir]
    python main.py input.jpg
    python main.py input.jpg ./results/
"""

import sys
import os
import cv2
import numpy as np


def detect_shadow_mask(image_path: str, output_dir: str = None) -> tuple:
    """
    Detects shadow regions using LAB + Otsu + morphological cleanup.
    Returns: (original_bgr, binary_mask, brightened_result)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    h, w = img.shape[:2]

    # --- LAB color space: L channel captures luminance/shadow well ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # --- Otsu threshold on L channel to find dark (shadow) regions ---
    _, mask = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Adaptive morphology: kernel scales with image size ---
    k = max(5, int(min(h, w) * 0.015))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Remove noise specks, then close gaps in shadow region
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- Soft feathering at shadow boundary (penumbra approximation) ---
    mask_soft = cv2.GaussianBlur(mask, (21, 21), 0)

    # --- Classical brightening via Retinex (for comparison baseline) ---
    l_f = l.astype(np.float32) + 1.0
    log_l = np.log(l_f)
    blur_l = cv2.GaussianBlur(log_l, (101, 101), 0)
    retinex = (log_l - blur_l - np.mean(log_l - blur_l)) * 30 + 128
    retinex = np.clip(retinex, 0, 255).astype(np.uint8)

    alpha = 0.55
    mask_f = mask_soft.astype(np.float32) / 255.0
    l_out = np.clip(
        l * (1 - mask_f) + (alpha * retinex + (1 - alpha) * l) * mask_f, 0, 255
    ).astype(np.uint8)

    lab_out = cv2.merge((l_out, a, b))
    brightened = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # --- Save outputs ---
    if output_dir is None:
        output_dir = os.path.dirname(image_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    mask_path = os.path.join(output_dir, "shadow_mask.png")
    brightened_path = os.path.join(output_dir, "classical_brightened.jpg")

    cv2.imwrite(mask_path, mask)              # binary mask for ShadowFormer
    cv2.imwrite(brightened_path, brightened)  # classical result (baseline)

    print(f"[Stage 1] Shadow mask saved   -> {mask_path}")
    print(f"[Stage 1] Classical result    -> {brightened_path}")

    return img, mask, brightened, mask_path


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python main.py <image_path> [output_dir]")
    #     sys.exit(1)

    img_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\college\2nd YEAR_SEM-4\IP\shadow_removal\test_images\99-4.png"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(img_path) or "."

    print(f"Processing image: {img_path}")
    print(f"Output directory: {out_dir}")

    original, mask, result, mpath = detect_shadow_mask(img_path, out_dir)

    cv2.imshow("Original", original)
    cv2.imshow("Shadow Mask (binary)", mask)
    cv2.imshow("Classical Brightened (baseline)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()