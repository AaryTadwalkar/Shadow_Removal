# Shadow Removal: From Classical IP → Full Regeneration Pipeline

## The Core Problem You Identified

Your current `classical_IP/main.py` does:
1. ✅ Detects shadow mask (Otsu thresholding on L channel)
2. ✅ Brightens the shadow region (Retinex + soft alpha blend)
3. ❌ **Cannot regenerate texture/color** — the shadow just looks "lifted" and unnatural

You are 100% correct: classical IP can *compensate* for shadows but cannot *inpaint* them.

---

## The Solution: A 3-Stage Hybrid Pipeline

```
Input Image
    │
    ▼
[Stage 1] SHADOW DETECTION  (Classical IP — what you already have)
    - LAB color space + Otsu thresholding
    - Morphological cleanup
    - Produces: binary shadow mask
    │
    ▼
[Stage 2] SHADOW REMOVAL / INPAINTING  (Deep Learning — NEW)
    Choose one path (see below):
    Path A: ShadowFormer  → end-to-end transformer
    Path B: LaMa Inpainting → general inpainting on masked region
    │
    ▼
[Stage 3] POST-PROCESSING  (Classical IP — polish)
    - CLAHE for local contrast
    - Edge-aware feathering at shadow boundary
    - Produces: final output
    │
    ▼
Output: Shadow-Free Image
```

---

## The Two Implementation Paths

### Path A — ShadowFormer (RECOMMENDED for academics)
**Paper:** "ShadowFormer: Global Context Helps Image Shadow Removal" — AAAI 2023  
**Approach:** End-to-end transformer trained specifically for shadow removal on ISTD dataset  
**Pros:** State-of-the-art PSNR/SSIM, purpose-built for shadows, gives great academic citations  
**Cons:** Needs pretrained weights (~200MB), cloning the repo, slightly complex setup

### Path B — Classical Mask + LaMa Inpainting (EASIER for demo)
**Tool:** `simple-lama-inpainting` (pip install) — LaMa is a Large Mask Inpainting model  
**Approach:** You detect the mask with classical IP (Stage 1), then LaMa *regenerates* the shadow region from context  
**Pros:** Single pip install, works out of the box, no custom model code  
**Cons:** Not shadow-specific (generic inpainting), weaker academic citation, may hallucinate textures

> [!IMPORTANT]
> For your **Image Processing course project**, I recommend **Path A** (ShadowFormer) because:
> - It maps directly to published AAAI/CVPR research
> - You can compare your classical IP results vs. ShadowFormer results quantitatively
> - It makes a much stronger project narrative: "We showed classical methods are insufficient and implemented a transformer-based solution"
>
> BUT: If ShadowFormer setup is blocked (GPU issues, dependency hell), **Path B with LaMa is your fallback** and still demonstrates the inpainting concept clearly.

---

## Open Questions

> [!IMPORTANT]
> **Do you have a GPU (NVIDIA CUDA)?** ShadowFormer runs significantly faster on GPU. On CPU it is slow (~30-120 seconds per image) but still works.

> [!WARNING]
> **Do you need to train anything?** NO. Both paths use pretrained weights. This is pure **inference** (testing a pre-built model on your own photos). Your course project will be about *integrating* these into a coherent pipeline.

---

## Proposed Project Structure

```
shadow_removal/
├── classical_IP/
│   ├── main.py               ← already done (keep as Stage 1)
│   ├── input.jpg
│   └── ...output images...
│
├── deep_learning/
│   ├── ShadowFormer/          ← cloned from GitHub (Path A)
│   │   ├── ... (repo files)
│   │   └── checkpoints/       ← pretrained .pth weights
│   │
│   └── remove.py              ← YOUR script: wraps ShadowFormer for single-image inference
│
├── hybrid_pipeline/
│   ├── pipeline.py            ← Full pipeline: Stage1 + Stage2 + Stage3
│   ├── evaluate.py            ← PSNR / SSIM metrics
│   └── results/               ← side-by-side comparison images
│
├── test_images/               ← your test inputs (outdoor + indoor shadows)
├── memory-bank/               ← project docs
└── README.md
```

---

## Proposed Changes

### Stage 1 — Improve Classical IP (already done, minor improvements)

#### [MODIFY] [main.py](file:///c:/college/2nd%20YEAR_SEM-4/IP/shadow_removal/classical_IP/main.py)
- Fix the hardcoded absolute path — make it take `sys.argv[1]`
- Add: adaptive kernel morphology (shadow size-dependent)
- Add: penumbra detection (gradient-based soft edges around shadow boundary)
- Save mask as `shadow_mask.png` for use by Stage 2

---

### Stage 2 — Deep Learning Shadow Removal

#### [NEW] `deep_learning/` folder

**Path A: ShadowFormer wrapper**

Steps:
1. `git clone https://github.com/GuoLanqing/ShadowFormer` into `deep_learning/ShadowFormer/`
2. Download pretrained weights from the repo's release page (ISTD checkpoint)
3. Create `deep_learning/remove.py` — a clean wrapper script that:
   - Accepts `input_image_path` and `shadow_mask_path`
   - Runs ShadowFormer inference
   - Returns and saves shadow-free image

**Path B: LaMa Inpainting (fallback)**

```python
# deep_learning/remove_lama.py
from simple_lama_inpainting import SimpleLama
from PIL import Image

def remove_shadow_lama(image_path, mask_path, output_path):
    model = SimpleLama()
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    result = model(img, mask)
    result.save(output_path)
```

Install: `pip install simple-lama-inpainting`

---

### Stage 3 — Hybrid Pipeline + Evaluation

#### [NEW] `hybrid_pipeline/pipeline.py`
Full orchestration:
```
1. Run Stage 1 → get shadow mask
2. Run Stage 2 → get inpainted image
3. Run post-processing → CLAHE + feathering
4. Save comparison grid
```

#### [NEW] `hybrid_pipeline/evaluate.py`
Metrics (if you have ground truth shadow-free images from ISTD dataset):
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity)
- MAE (Mean Absolute Error in shadow region only)

---

## Execution Order

| Step | Task | Time Estimate |
|------|------|---------------|
| 1 | Improve `classical_IP/main.py` (fix path, save mask) | 30 min |
| 2 | Clone ShadowFormer repo | 5 min |
| 3 | Download pretrained weights | 10 min |
| 4 | Write `deep_learning/remove.py` | 1-2 hr |
| 5 | Write `hybrid_pipeline/pipeline.py` | 1 hr |
| 6 | Write `hybrid_pipeline/evaluate.py` | 30 min |
| 7 | Run on test images, compare results | 1 hr |
| 8 | Create report / viva guide | 1-2 hr |

**Total: ~1-2 days of focused work**

---

## Academic Framing (for your viva/report)

| Component | Technique | Why Used |
|-----------|-----------|----------|
| Shadow Detection | LAB + Otsu Thresholding + Morphology | Classical IP — interpretable, no data needed |
| Shadow Removal | ShadowFormer (Transformer) | DL — learns illumination relationships globally |
| Post-processing | CLAHE + Edge Feathering | Classical IP — fixes local contrast, smooth boundaries |
| Evaluation | PSNR, SSIM, MAE | Standard benchmark metrics (ISTD dataset) |

**Key insight to present:** Classical IP can detect and brighten but **cannot regenerate** — deep learning is required for true shadow removal because shadows alter not just brightness but also texture, color temperature, and surface appearance.

---

## Verification Plan

### Code Testing
- Run pipeline on `classical_IP/input.jpg` (already have it)
- Download 5-10 images from ISTD test set for benchmark

### Metrics
```bash
python hybrid_pipeline/evaluate.py --input test_images/ --gt ground_truth/
```

### Visual Comparison
- Side-by-side grid: `Original | Classical Result | ShadowFormer Result | Ground Truth`



### Implementation Plan Assessment
The plan is conceptually correct but has one mismatch with your actual architecture from our previous work:
Plan says Stage 1 = classical Otsu detection only.
Your real architecture = SegFormer-B2 for detection + classical IP steps as load-bearing components.
That hybrid approach is strictly better — keep it. Here's the corrected picture:
Input
  │
  ▼
[Stage 1] DETECTION — SegFormer-B2 (fine-tuned on ISTD)
          + Classical load-bearing steps:
            HSV/LAB masking → Otsu → Morphological cleanup
            → Watershed with Sobel gradients
          → binary shadow mask
  │
  ▼
[Stage 2] REMOVAL — ShadowFormer
          Input: img (0–1 normalized) + mask (0–1)
          Call: model(img_t, mask_t)
          Output: ×255 denormalized
  │
  ▼
[Stage 3] POST-PROCESSING — Classical
          CLAHE → Gaussian feathered blend at boundary
  │
  ▼
Output

Specific Betterments to the Plan
1. The two bugs you already identified — confirm these are fixed before anything else:

SegFormer: must load fine-tuned .pth checkpoint, not random weights
ShadowFormer: normalize input to [0,1], not [0,255]; denormalize output by ×255

2. Your RTX 4050 6GB is sufficient — ShadowFormer-B (ISTD checkpoint) fits in ~3GB VRAM. SegFormer-B2 is another ~1GB. You have headroom.
3. Drop LaMa entirely — with your GPU and ShadowFormer available, LaMa is unnecessary. It's generic inpainting and will produce hallucinated grass texture. ShadowFormer was trained on ISTD outdoor grass images — it's the right tool for your exact input image.
4. The classical steps must remain load-bearing for your examiner — don't let them become cosmetic. The defensible framing is:

Otsu/morphology: produces the coarse mask that seeds SegFormer's attention
Watershed + Sobel: refines shadow boundary (penumbra region)
CLAHE: corrects local contrast in recovered region post-ShadowFormer
Gaussian feathering: smooth blend at mask edge

5. Evaluation — PSNR ≥ 28 dB and SSIM ≥ 0.85 on ISTD are achievable with ShadowFormer. Your V1 was ~20.8 dB / 0.756, so this is a meaningful improvement to show.

What to Do Next (in order)

Confirm ShadowFormer pipeline works on one ISTD outdoor test image — model(img_t, mask_t) with correct normalization, check output looks like lit grass
Confirm SegFormer-B2 loads fine-tuned weights and produces a clean binary mask on your input.jpg
Wire Stage 1 → Stage 2 → Stage 3 in pipeline.py
Run evaluate.py on ISTD test split for metrics