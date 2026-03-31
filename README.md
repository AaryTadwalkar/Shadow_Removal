# Shadow Removal Project

## Project Structure
```
shadow_removal/
├── classical_IP/
│   ├── main.py               ← Stage 1: shadow detection (run standalone)
│   ├── input.jpg             ← sample input
│   └── ...outputs...
│
├── deep_learning/
│   ├── remove.py             ← Stage 2: ShadowFormer inference wrapper
│   └── ShadowFormer/         ← cloned model repo
│       └── checkpoints/      ← PUT YOUR .pth WEIGHTS HERE
│
├── hybrid_pipeline/
│   ├── pipeline.py           ← Full end-to-end runner (recommended)
│   ├── evaluate.py           ← PSNR/SSIM/MAE metrics
│   └── results/              ← output images saved here
│
├── test_images/              ← PUT YOUR TEST IMAGES HERE
└── memory-bank/              ← project documentation
```

---

## ⚠️ STEP 1 — Download Pretrained Weights (REQUIRED)

Download the ISTD pretrained model from Google Drive:

**Link**: https://drive.google.com/file/d/1bHbkHxY5D5905BMw2jzvkzgXsFPKzSq4/view?usp=share_link

**Save as**: `deep_learning/ShadowFormer/checkpoints/ISTD.pth`

> The ISTD model was trained on indoor + outdoor shadow images — best for general use.

---

## STEP 2 — Add Test Images

Place your shadow images in `test_images/` folder.

---

## STEP 3 — Run the Full Pipeline

```powershell
cd "c:\college\2nd YEAR_SEM-4\IP\shadow_removal"

python hybrid_pipeline/pipeline.py test_images/your_image.jpg deep_learning/ShadowFormer/checkpoints/ISTD.pth
```

Results saved to `hybrid_pipeline/results/`:
- `shadow_mask.png` — detected shadow region
- `classical_brightened.jpg` — Stage 1 baseline
- `shadowformer_output.jpg` — Stage 2 deep learning result
- `final_output.jpg` — Stage 3 post-processed
- `comparison_grid.jpg` — 4-panel side-by-side

---

## Run Stage 1 Only (Classical IP)
```powershell
python classical_IP/main.py test_images/your_image.jpg
```

## Run Stage 2 Only (ShadowFormer)
```powershell
python deep_learning/remove.py test_images/your_image.jpg path/to/mask.png deep_learning/ShadowFormer/checkpoints/ISTD.pth
```

## Evaluate Metrics (needs ground truth)
```powershell
python hybrid_pipeline/evaluate.py --input hybrid_pipeline/results/final_output.jpg --gt path/to/ground_truth.jpg
```

---

## Academic Reference
**ShadowFormer**: Guo et al., "ShadowFormer: Global Context Helps Image Shadow Removal", AAAI 2023.  
Paper: https://arxiv.org/pdf/2302.01650.pdf

**Key Result**: ShadowFormer achieves **PSNR 32.21 dB, SSIM 0.968** on ISTD dataset  
(vs classical methods: ~27 dB PSNR)

# Shadow_Removal
course project for Image Processing

