# Shadow Removal Project — Memory Bank

## Project Brief
**Course**: Image Processing (2nd Year, Sem 4)  
**Title**: Detection and Removal of Shadows in Outdoor and Indoor Images  
**Goal**: Hybrid pipeline — Classical IP (shadow detection) + ShadowFormer DL (regeneration)

## System Architecture

```
Stage 1: classical_IP/main.py
    → LAB + Otsu + Morphology → shadow_mask.png + classical_brightened.jpg

Stage 2: deep_learning/remove.py
    → ShadowFormer (AAAI 2023) → shadowformer_output.jpg

Stage 3: hybrid_pipeline/pipeline.py (runs all stages)
    → CLAHE post-process → final_output.jpg + comparison_grid.jpg

Evaluation: hybrid_pipeline/evaluate.py
    → PSNR, SSIM, MAE metrics
```

## Tech Stack
- **Python 3.13** (system)
- **PyTorch 2.6.0+cu124** — CUDA 12.4, RTX 4050 6GB
- **OpenCV** — classical IP
- **ShadowFormer** — AAAI 2023, cloned at `deep_learning/ShadowFormer/`
- **einops, timm, scikit-image** — ShadowFormer dependencies

## Key Files
| File | Purpose |
|------|---------|
| `classical_IP/main.py` | Stage 1: shadow mask generation |
| `deep_learning/remove.py` | Stage 2: ShadowFormer inference wrapper |
| `deep_learning/ShadowFormer/` | Cloned model repo |
| `deep_learning/ShadowFormer/checkpoints/` | Place .pth weights here |
| `hybrid_pipeline/pipeline.py` | Full end-to-end runner |
| `hybrid_pipeline/evaluate.py` | PSNR/SSIM/MAE metrics |
| `test_images/` | Put your test photos here |

## Status (as of 2026-05-14) — PIPELINE WORKING ✅
- [x] Classical IP main.py — rewritten, takes CLI args, saves mask
- [x] ShadowFormer repo cloned
- [x] CUDA PyTorch 2.6+cu124 installed + verified (RTX 4050 Laptop GPU)
- [x] Model instantiation tested (11.4M params, ShadowFormer)
- [x] scipy.misc deprecated import fixed in model.py
- [x] Unicode arrow print fix for Windows cp1252 in all scripts
- [x] ISTD.pth weights downloaded (137MB)
- [x] Full pipeline run SUCCESS on test_images/input.jpg
- [x] Verified: shadow brightness 26.5 -> 60.5 (+34 pts) in shadow region
- [x] All 5 outputs generated in hybrid_pipeline/results/

## Next Steps
1. User downloads ISTD pretrained weights → place in `deep_learning/ShadowFormer/checkpoints/ISTD.pth`
2. Place test image in `test_images/`
3. Run: `python hybrid_pipeline/pipeline.py test_images/your_image.jpg deep_learning/ShadowFormer/checkpoints/ISTD.pth`
