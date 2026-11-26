# MILK10k teacher (clinical + dermoscopic + metadata)

## Commands (in this order)

```bash
pip install -r requirements.txt
python check.py            # ~5 min. MUST say "ALL CHECKS PASSED" before training
python train.py            # all 5 folds (finished folds are skipped if you re-run)
python tune_thresholds.py  # OOF metrics + one threshold per class
python predict.py          # test set -> runs/teacher_convnext/submission.csv
```

Only file you edit: `config.yaml` (paths at the top). Everything else is fixed.

Faster: run folds in parallel on separate GPUs (after `check.py` passed):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --fold 0 &
CUDA_VISIBLE_DEVICES=1 python train.py --fold 1 &
...
```
The fold split is written to `work_dir/folds.csv` the first time and reused, so every run sees the same folds.

## Files

| File | What it is |
|---|---|
| `config.yaml` | all paths and settings |
| `model.py` | architecture (2× ConvNeXt, metadata MLP, FiLM + gated fusion, LayerNorm→Dropout→Linear head) |
| `data.py` | reads the CSVs, pairs clinical + dermoscopic image per lesion, encodes metadata, augmentation, folds |
| `engine.py` | training loop, EMA, metrics, thresholds |
| `train.py` | 5-fold training |
| `tune_thresholds.py` | combines out-of-fold predictions, per-class thresholds |
| `predict.py` | 5-fold ensemble on the test set, writes the submission |
| `check.py` | pre-flight test of everything on a tiny subset |

## What is implemented (as decided)

- **Inputs:** clinical image + dermoscopic image (separate ConvNeXt-Base @384, ImageNet-22k pretrained) + metadata.
- **Metadata:** age/100 + `age_missing` flag; sex, site, skin tone one-hot with an extra "unknown" slot;
  MONET scores of the clinical image (+ missing flag). The dermoscopic image's MONET scores go with the
  dermoscopic branch, so they are hidden together with the dermoscopic image.
- **Fusion:** FiLM (metadata scales/shifts both image features) + gate (weighs dermoscopic vs clinical).
- **Head:** LayerNorm → Dropout(0.3) → Linear(512→11).
- **Loss:** cross-entropy, class weights 1/√count (computed on each fold's training part), label smoothing 0.05.
- **Training:** 5-fold stratified, AdamW (backbone 1e-4, rest 1e-3), cosine schedule with 2 warm-up epochs,
  30 epochs, EMA weights, bf16, gradient clipping. Dermoscopic image dropped 20 % of the time.
- **Augmentation (train only):** random crop/zoom, flips, 90° rotations, light colour jitter.
- **Checkpoint:** best validation **macro-F1** per fold (EMA weights).
- **After training:** per-class thresholds tuned on the out-of-fold predictions; for the submission the
  probabilities are re-mapped so each class's threshold sits at the challenge's 0.5 cut-off.
- **Saved for the student (distillation):** `oof_teacher_outputs.npz` = for every training lesion its
  out-of-fold logits, probabilities, features `f_clin`, `f_derm`, fused `h`, gate `g`, and the logits the
  teacher gives *without* the dermoscopic image.

## Outputs (in `work_dir`)

`fold*/best.pt`, `fold*/log.csv`, `fold*/oof.npz`, `folds.csv`, `meta_encoder.json`,
`thresholds.json`, `oof_metrics.json`, `oof_per_class.csv`, `oof_predictions.csv`,
`oof_teacher_outputs.npz`, `submission.csv`, `test_probs_raw.csv`, `test_teacher_outputs.npz`.

## Notes

- Column names are detected automatically; `check.py` prints what it found. If it can't find one,
  set it under `data.columns` in `config.yaml`.
- Images are matched by file name = the image id column (e.g. `ISIC_1234567.jpg`), searched in all sub-folders.
- Memory: batch 16 at 384 px with two ConvNeXt-Base fits easily on a large GPU; `check.py` prints the
  peak memory. If it runs out, set `batch_size: 8`.
- MILK10k has no patient ids, so folds are split by lesion.
