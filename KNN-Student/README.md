# KNN-Student

A student model that diagnoses a skin lesion from **only the clinical photo and the patient's
metadata**. It needs no dermoscopic image. It learns from the trained `Fusion-Teacher` (knowledge
distillation) and makes its final decision with a **k-nearest-neighbour (kNN) vote**: it looks
up the 20 most similar lesions from training and lets them vote.

The teacher is only used during training. At test time the student works on its own.

| | Inputs | Out-of-fold macro-F1 | AUC |
|---|---|---|---|
| Fusion-Teacher | clinical + dermoscopic + metadata | 0.578 | 0.913 |
| Teacher with the dermoscopic image hidden (the starting point) | clinical + metadata | 0.481 | 0.863 |
| **KNN-Student, kNN decision (final)** | clinical + metadata | **0.493** | 0.770 |
| KNN-Student, classification head | clinical + metadata | 0.495 | 0.878 |

## The model

```
clinical photo ──► ConvNeXt-Base ──► clinical feature ──┐
                                                        │
"no dermoscopy" vector (learned constant) ──────────────┤
                                                        ▼
metadata ────────► small MLP ──────► metadata feature ──► FiLM + gated fusion ──► fused feature h (512)
                                                                                     │
                                                         ┌───────────────────────────┤
                                                         ▼                           ▼
                                                classification head         kNN vote over the
                                                 (training signal)            case bank (final)
```

- **It is the teacher with the dermoscopic branch removed.** Every layer is copied from the
  teacher of the same fold. The dermoscopic encoder is dropped, and its slot in the fusion gets the
  teacher's learned "no dermoscopy" vector. So at the start of training the student is exactly the
  teacher with the dermoscopic image hidden.
- **Case bank.** After training, the fused feature `h` and the diagnosis of every training lesion
  in the fold are stored in `bank.npz` (about 4,200 lesions).
- **kNN decision.** For a new lesion, the 20 most similar stored lesions (by cosine similarity)
  vote. Closer lesions get more weight, and rare classes get a boost (weight 1/√count).
- **Size.** 89.0 M parameters, half the teacher's, because there is only one ConvNeXt-Base.

## How it learns from the teacher

Each student fold learns from the frozen teacher of the same fold. During training the teacher
still sees all three inputs, but the student sees only the clinical photo and the metadata. The
loss has three parts:

| Loss | What it does |
|---|---|
| Cross-entropy | predict the true diagnosis |
| Logit distillation (KL, temperature 3) | match the teacher's class scores |
| Feature matching (cosine) | match the teacher's fused feature `h` |

All three use the same class weights, 1/√count.

## Code structure

```
KNN-Student/
├── config.yaml          all paths and settings (the only file you need to edit)
├── student_model.py     the student network + copying the weights from the teacher
├── student_data.py      clinical-only dataset (never opens a dermoscopic image)
├── student_engine.py    frozen teacher, distillation loss, training loop, kNN case bank
├── train.py             trains the 5 folds, builds each fold's case bank
├── evaluate.py          validation results: teacher vs. student head vs. student kNN
├── predict.py           5-fold ensemble on the test set, writes the submissions
├── sweep.py             (optional) tries different kNN settings on the saved banks
├── data.py, engine.py, model.py   copies of the teacher's files (needed to load the teacher)
└── requirements.txt
```

## How to train

### 1. Train the teacher first

The student is built from the teacher's weights, so you need a fully trained `Fusion-Teacher`
(all 5 folds), or its downloaded checkpoints.

### 2. Install

```bash
pip install -r requirements.txt
```

You need a GPU. One fold takes about 35 minutes on ours (15 epochs), so all 5 folds take about 3
hours.

### 3. Set the paths in `config.yaml`

Point it to the teacher and to the data. The data paths are the same as in the teacher's config:

```yaml
teacher_work_dir: /path/to/Fusion-Teacher/runs/teacher_convnext   # the trained teacher

data:
  train_meta_csv: /path/to/MILK10k/MILK10k_Training_Metadata.csv
  train_gt_csv:   /path/to/MILK10k/MILK10k_Training_GroundTruth.csv
  train_img_dir:  /path/to/MILK10k/MILK10k_Training_Input
  test_meta_csv:  /path/to/MILK10k/MILK10k_Test_Metadata.csv
  test_img_dir:   /path/to/MILK10k/MILK10k_Test_Input

work_dir: runs/student_knn            # where all outputs are saved
```

Do not change the `model` block or `folds` / `seed` / `img_size`. They must match the teacher.

### 4. Run

```bash
python train.py       # trains folds 0-4 (folds that are already finished are skipped)
python evaluate.py    # validation table: teacher / student head / student kNN
python predict.py     # test set -> runs/student_knn/submission_knn_tau0.25.csv
```

If you have several GPUs, you can train the folds in parallel:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --fold 0 &
CUDA_VISIBLE_DEVICES=1 python train.py --fold 1 &
# ... one line per fold, then run evaluate.py and predict.py
```

**Changing the kNN settings** (`k`, `temperature` in the `knn` block) **does not need retraining.**
`evaluate.py` and `predict.py` rebuild the decision from the saved case banks. To try many settings
at once (CPU only, about a minute):

```bash
python sweep.py       # -> runs/student_knn/sweep.csv
```

### Training settings

| Setting | Value |
|---|---|
| Starting weights | the teacher of the same fold |
| Split | the teacher's 5 folds (`folds.csv` is reused) |
| Epochs / batch size | 15 / 16 |
| Optimizer | AdamW, learning rate 5e-5 (encoder) and 5e-4 (rest), weight decay 0.05 |
| Schedule | cosine, with 1 warm-up epoch |
| Other | EMA weights (0.998), bf16, gradient clipping 1.0, flip test-time augmentation |
| Checkpoint kept | the last epoch (no validation labels are used to pick it) |
| kNN | k = 20, temperature 0.07 |
| Post-processing | logits − 0.25 · log(class frequency), then the top class |

All of these are in `config.yaml`.

## Outputs

Everything is written to `work_dir` (`runs/student_knn/`):

```
runs/student_knn/
├── fold0/ ... fold4/
│   ├── best.pt                       student weights of this fold
│   ├── bank.npz                      case bank: feature h + diagnosis of each training lesion
│   ├── log.csv                       losses (ce / kd / feat) and validation macro-F1 per epoch
│   └── oof.npz                       this fold's predictions on its validation lesions
├── folds.csv                         copy of the teacher's split
├── meta_encoder.json                 copy of the teacher's metadata encoding
│
├── oof_metrics.json                  validation results: teacher, student head, student kNN
├── oof_per_class.csv                 F1 per class
├── oof_student_outputs.npz           validation logits + kNN votes for every training lesion
│
├── submission_knn_tau0.25.csv        FINAL test predictions (kNN decision)
├── submission_head_tau0.25.csv       test predictions of the classification head (for comparison)
└── test_student_outputs.npz          test logits + kNN votes
```

`sweep.csv` also appears if you run `sweep.py`.

The submissions have one row per test lesion and one column per class. The predicted class has a
value of 0.5 or more, and all others are below 0.5.

The `.pt` and `.npz` files are not in this git repo because they are too large. Download them from
the checkpoint link in the main README.

## Notes

- **No dermoscopy leaks into the student.** The dermoscopic image is passed only to the frozen
  teacher during training. Validation, the case bank and the test predictions use a dataset that
  never opens a dermoscopic file. Each fold's case bank holds only that fold's training lesions.
- **The teacher folder is only read.** Nothing is written into it.
- **Running out of GPU memory?** Set `batch_size: 8` in `config.yaml`.
- **Slightly optimistic validation numbers.** The teacher's checkpoints were picked by validation
  macro-F1, so the student's out-of-fold numbers are a little high as well. The hidden test set is
  the clean comparison.
