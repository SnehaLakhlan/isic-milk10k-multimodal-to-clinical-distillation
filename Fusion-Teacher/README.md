# Fusion-Teacher

The teacher model of this project. For each skin lesion it looks at **three inputs together**: a
clinical close-up photo, a dermoscopic image and the patient's metadata. It then predicts one of
the 11 MILK10k diagnoses (AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC).

Because it sees the dermoscopic image, the teacher is the strongest model in the project. The two
students (`KNN-Student/`, `Halluc-Student/`) are trained to copy it while seeing only the clinical
photo and the metadata.

| | Out-of-fold macro-F1 | AUC |
|---|---|---|
| Teacher (clinical + dermoscopic + metadata) | **0.578** | 0.913 |
| Same teacher with the dermoscopic image hidden | 0.481 | 0.863 |

## The model

```
clinical photo ─────► ConvNeXt-Base ──► clinical feature ─────┐
                                                              │
dermoscopic image ──► ConvNeXt-Base ──► dermoscopic feature ──┤
                                                              ▼
metadata ───────────► small MLP ──────► metadata feature ──► FiLM + gated fusion ──► fused feature (512)
(age, sex, site, skin tone,                                                                │
 MONET concept scores)                                                                     ▼
                                                                         classification head ──► 11 classes
```

- **Two image encoders.** ConvNeXt-Base at 384 px, pretrained on ImageNet-22k, with separate
  weights for the clinical photo and the dermoscopic image.
- **FiLM.** The metadata scales and shifts both image features.
- **Gate.** For each lesion, the model learns how much to trust the dermoscopic feature versus the
  clinical one.
- **Missing dermoscopy.** If a lesion has no dermoscopic image, a learned "missing" vector takes
  its place. During training the dermoscopic image is dropped 20 % of the time on purpose, so the
  model also learns to work without it.
- **Size.** 177 M parameters: two ConvNeXt-Base encoders of 87.6 M each, plus about 2 M for the
  fusion and the head.

## Code structure

```
Fusion-Teacher/
├── config.yaml           all paths and settings (the only file you need to edit)
├── data.py               reads the CSVs, pairs each lesion's clinical + dermoscopic image,
│                         encodes metadata, augmentation, 5-fold split
├── model.py              the network above
├── engine.py             training loop, EMA, metrics, threshold tuning
├── train.py              trains the 5 folds
├── tune_thresholds.py    combines the 5 folds' validation predictions, tunes one threshold per class
├── predict.py            5-fold ensemble on the test set, writes the submission
├── predict_clin_only.py  (optional) the same, but with the dermoscopic images hidden
├── posthoc_adjust.py     (optional) a different decision rule (logit adjustment with tau)
└── requirements.txt
```

## How to train

### 1. Install

```bash
pip install -r requirements.txt
```

You need a GPU. One fold takes about 50 minutes on ours (30 epochs), so all 5 folds take about 4
hours.

### 2. Get the data

Download MILK10k (see the main README). You need these five files and folders:

```
MILK10k_Training_Metadata.csv     MILK10k_Training_Input/
MILK10k_Training_GroundTruth.csv  MILK10k_Test_Input/
MILK10k_Test_Metadata.csv
```

### 3. Set the paths in `config.yaml`

Edit the `data` block at the top so it points to your copy of the data:

```yaml
data:
  train_meta_csv: /path/to/MILK10k/MILK10k_Training_Metadata.csv
  train_gt_csv:   /path/to/MILK10k/MILK10k_Training_GroundTruth.csv
  train_img_dir:  /path/to/MILK10k/MILK10k_Training_Input
  test_meta_csv:  /path/to/MILK10k/MILK10k_Test_Metadata.csv
  test_img_dir:   /path/to/MILK10k/MILK10k_Test_Input

work_dir: runs/teacher_convnext     # where all outputs are saved
```

That is all you have to change. The column names are detected automatically. If one is not found,
`train.py` stops and tells you which, and you can set it under `data.columns`.

### 4. Run

```bash
python train.py              # trains folds 0-4 (folds that are already finished are skipped)
python tune_thresholds.py    # validation metrics + one threshold per class
python predict.py            # test set -> runs/teacher_convnext/submission.csv
```

If you have several GPUs, you can train the folds in parallel:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --fold 0 &
CUDA_VISIBLE_DEVICES=1 python train.py --fold 1 &
# ... one line per fold, then run tune_thresholds.py and predict.py as above
```

Optional, to get the teacher's predictions **without** dermoscopic images (the baseline the students
are compared against):

```bash
python predict_clin_only.py  # -> runs/teacher_convnext/submission_clin_only.csv
```

### Training settings

| Setting | Value |
|---|---|
| Split | 5-fold, stratified by diagnosis, split by lesion |
| Epochs / batch size | 30 / 16 |
| Optimizer | AdamW, learning rate 1e-4 (encoders) and 1e-3 (rest), weight decay 0.05 |
| Schedule | cosine, with 2 warm-up epochs |
| Loss | cross-entropy, class weights 1/√count, label smoothing 0.05 |
| Augmentation | random crop and zoom, flips, 90° rotations, light colour jitter |
| Other | EMA weights (0.998), bf16, gradient clipping 1.0, flip test-time augmentation |
| Checkpoint kept | the epoch with the best validation macro-F1, per fold |

All of these are in `config.yaml`.

## Outputs

Everything is written to `work_dir` (`runs/teacher_convnext/`):

```
runs/teacher_convnext/
├── fold0/ ... fold4/
│   ├── best.pt                       model weights of this fold
│   ├── log.csv                       loss and validation macro-F1 per epoch
│   └── oof.npz                       this fold's predictions on its validation lesions
├── folds.csv                         which lesion is in which fold (the students reuse it)
├── meta_encoder.json                 how the metadata is encoded (needed for prediction)
│
├── oof_metrics.json                  validation results: macro-F1, balanced accuracy, AUC, ...
├── oof_per_class.csv                 F1 per class
├── oof_predictions.csv               validation probabilities for every training lesion
├── thresholds.json                   one decision threshold per class
├── oof_teacher_outputs.npz           logits + features for every training lesion (used by the students)
│
├── submission.csv                    FINAL test predictions
├── test_probs_raw.csv                test probabilities before the thresholds
├── test_teacher_outputs.npz          test logits + features
│
├── submission_clin_only.csv          test predictions without dermoscopy   (predict_clin_only.py)
├── thresholds_clin_only.json         thresholds for that case              (predict_clin_only.py)
└── test_teacher_outputs_clin_only.npz                                      (predict_clin_only.py)
```

`submission.csv` has one row per test lesion and one column per class. Each class's threshold is
mapped to 0.5, so a value of 0.5 or more means that class is predicted.

The `.pt` and `.npz` files are not in this git repo because they are too large. Download them from
the checkpoint link in the main README.

## Notes

- **The students depend on this folder.** They read `fold*/best.pt`, `folds.csv` and
  `meta_encoder.json` from here, through `teacher_work_dir` in their own `config.yaml`.
- **Running out of GPU memory?** Set `batch_size: 8` in `config.yaml`.
- **Images** are found by file name (the `isic_id` column, e.g. `ISIC_1234567.jpg`) anywhere inside
  the image folder.
- **No patient IDs.** MILK10k has none, so the folds are split by lesion.
- **Slightly optimistic validation numbers.** Each fold's checkpoint is chosen by its best
  validation macro-F1, so the out-of-fold numbers are a little high. The hidden test set is the
  clean comparison.
