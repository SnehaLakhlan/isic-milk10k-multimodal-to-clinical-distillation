# Halluc-Student

A student model that diagnoses a skin lesion from **only the clinical photo and the patient's
metadata**. It needs no dermoscopic image. Like `KNN-Student`, it learns from the trained
`Fusion-Teacher` (knowledge distillation). It adds a **hallucinator**: a small network that
predicts, from the photo, the dermoscopic feature the teacher would have seen.

The teacher is only used during training. At test time the student works on its own. This is
the best student in the project.

| | Inputs | Out-of-fold macro-F1 | AUC |
|---|---|---|---|
| Fusion-Teacher | clinical + dermoscopic + metadata | 0.578 | 0.913 |
| Teacher with the dermoscopic image hidden (the starting point) | clinical + metadata | 0.481 | 0.863 |
| **Halluc-Student (final)** | clinical + metadata | **0.504** | 0.874 |
| Same student, no distillation (`no_kd`) | clinical + metadata | 0.466 | 0.810 |

Without the teacher, the student ends up below its starting point (0.466). With distillation and
the hallucinator it reaches 0.504, which closes 23 % of the gap to the teacher with dermoscopy.

## The model

```
clinical photo ──► ConvNeXt-Base ──► clinical feature ──┬───────────────────────────────┐
                                                        │                               │
metadata ────────► small MLP ──────► metadata feature ──┼──► HALLUCINATOR ──► predicted │
                                                        │    (codebook + MLP)   dermo-  │
                                                        │                       scopic  │
                                                        │                       feature │
                                                        ▼                         │     │
                                         FiLM + gated fusion ◄────────────────────┘◄────┘
                                                  │
                                                  ▼
                                      fused feature h (512) ──► classification head ──► 11 classes
```

- **It starts as the teacher with the dermoscopic branch removed.** Every layer is copied from
  the teacher of the same fold, and the dermoscopic encoder is dropped.
- **Hallucinator.** In the slot where the teacher's fusion expects the dermoscopic feature, the
  student puts a *prediction* of it:
  - **Codebook:** before training, K-means over the teacher's real dermoscopic features (training
    lesions of that fold only) gives 32 typical "dermoscopic patterns". They are fixed and never
    trained.
  - **MLP:** a small network reads the clinical feature and the metadata and decides how to mix
    those 32 patterns.
  - **Start:** it outputs the teacher's "no dermoscopy" vector, so at step 0 the student is
    exactly the teacher with the dermoscopic image hidden.
- **Why it helps.** At test time the hallucinator cannot add information the photo doesn't
  contain. Its value is as a **training signal**: to predict the dermoscopic feature, the encoder
  has to keep fine photo detail that a plain class label would let it throw away.
- **Final decision.** The classification head, then logit adjustment.
- **Size.** 89.6 M parameters. The hallucinator adds only 0.6 M on top of one ConvNeXt-Base.

## How it learns from the teacher

Each student fold learns from the frozen teacher of the same fold. During training the teacher
still sees all three inputs, but the student sees only the clinical photo and the metadata. The
loss has four parts:

| Loss | What it does |
|---|---|
| Cross-entropy | predict the true diagnosis |
| Logit distillation (KL, temperature 3) | match the teacher's class scores |
| Feature matching (cosine) | match the teacher's fused feature `h` |
| **Hallucination (cosine)** | match the teacher's real dermoscopic feature |

All four use the same class weights, 1/√count. Lesions without a dermoscopic image are skipped by
the hallucination loss.

## Code structure

```
Halluc-Student/
├── config.yaml          all paths and settings (the only file you need to edit)
├── student_model.py     the student network + hallucinator + copying the weights from the teacher
├── student_data.py      clinical-only dataset (never opens a dermoscopic image)
├── student_engine.py    frozen teacher, the four losses, training loop, codebook fitting
├── train.py             trains the 5 folds
├── evaluate.py          validation table: the teacher (with / without dermoscopy) vs. the student runs
├── predict.py           5-fold ensemble on the test set, writes the submission
├── tune_decision.py     (optional) tunes one decision offset per class instead of tau
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

You need a GPU. One fold takes about 35–55 minutes on ours (15 epochs), so all 5 folds take about
3–4 hours.

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

work_dir: runs/halluc                 # where all outputs are saved
```

Do not change the `model` block or `folds` / `seed` / `img_size`. They must match the teacher.

### 4. Run

```bash
python train.py       # trains folds 0-4 (folds that are already finished are skipped)
python evaluate.py    # validation table next to the teacher
python predict.py     # test set -> runs/halluc/submission_tau0.25.csv
```

If you have several GPUs, you can train the folds in parallel:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --fold 0 &
CUDA_VISIBLE_DEVICES=1 python train.py --fold 1 &
# ... one line per fold, then run evaluate.py and predict.py
```

### The no-distillation control

To show what the teacher adds, train the same student with cross-entropy only. It goes to its own
folder, so nothing is overwritten:

```bash
python train.py --kd_weight 0 --feat_weight 0 --halluc_weight 0 --work_dir runs/no_kd
python evaluate.py --runs runs/halluc,runs/no_kd     # both runs in one table
```

### Training settings

| Setting | Value |
|---|---|
| Starting weights | the teacher of the same fold |
| Split | the teacher's 5 folds (`folds.csv` is reused) |
| Epochs / batch size | 15 / 16 |
| Optimizer | AdamW, learning rate 5e-5 (encoder) and 5e-4 (rest, incl. the hallucinator), weight decay 0.05 |
| Schedule | cosine, with 1 warm-up epoch |
| Hallucinator | codebook of 32 patterns, MLP with 512 hidden units, dropout 0.1 |
| Other | EMA weights (0.998), bf16, gradient clipping 1.0, flip test-time augmentation |
| Checkpoint kept | the last epoch (no validation labels are used to pick it) |
| Post-processing | logits − 0.25 · log(class frequency), then the top class |

All of these are in `config.yaml`. Any loss weight can also be set from the command line, e.g.
`--halluc_weight 0`.

## Outputs

Everything is written to `work_dir` (`runs/halluc/`):

```
runs/halluc/
├── fold0/ ... fold4/
│   ├── best.pt                  student weights of this fold (the codebook is stored inside)
│   ├── log.csv                  losses (ce / kd / feat / halluc) and validation macro-F1 per epoch
│   └── oof.npz                  this fold's validation predictions, fused feature h and predicted token x
├── folds.csv                    copy of the teacher's split
├── meta_encoder.json            copy of the teacher's metadata encoding
├── settings.json                the exact loss weights and hallucinator settings of this run
│
├── oof_metrics.json             validation results: teacher (with / without dermoscopy), halluc, no_kd
├── oof_per_class.csv            F1 per class for each of those
│
├── submission_tau0.25.csv       FINAL test predictions
└── test_outputs.npz             test logits
```

`runs/no_kd/` has the same layout for the no-distillation control. If you run `tune_decision.py`,
`decision_offsets.json` also appears.

The submission has one row per test lesion and one column per class. The predicted class has a
value of 0.5 or more, and all others are below 0.5.

The `.pt` and `.npz` files are not in this git repo because they are too large. Download them from
the checkpoint link in the main README.

## Notes

- **No dermoscopy leaks into the student.** The dermoscopic image is passed only to the frozen
  teacher during training. Validation and the test predictions use a dataset that never opens a
  dermoscopic file. The codebook, the class weights and the class frequencies all come from that
  fold's training lesions only.
- **The teacher folder is only read.** Nothing is written into it.
- **Running out of GPU memory?** Set `batch_size: 8` in `config.yaml`.
- **Slightly optimistic validation numbers.** The teacher's checkpoints were picked by validation
  macro-F1, so the student's out-of-fold numbers are a little high as well. The hidden test set is
  the clean comparison.
