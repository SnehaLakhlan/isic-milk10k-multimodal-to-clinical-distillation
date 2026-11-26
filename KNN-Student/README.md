# MILK10k student (clinical image + metadata only)

A camera-only student distilled from the trained teacher. At test time it gets the clinical photo
and the metadata, nothing else. The final decision is made by a kNN vote over a bank of past
clinical cases (the one classical module); a neural head exists only as a training signal.

## Setup

The folder is self-contained: all code it needs is inside. It only reads two things from outside,
both set at the top of `config.yaml`:

- `teacher_work_dir`: the trained teacher's run folder (`fold0..4/best.pt`, `folds.csv`,
  `meta_encoder.json`). The student is built from these weights and re-uses these folds.
- `data`: the MILK10k paths (same as in the teacher's config).

Put the folder anywhere, check those paths, and run the commands from inside it. The `model`
block and `folds` / `seed` / `img_size` in `config.yaml` are copies of the teacher's settings and
must stay identical to them.

## Commands (in this order, from inside this folder)

```bash
python check.py      # ~5 min. MUST say "ALL CHECKS PASSED" before training
python train.py      # all 5 folds (finished folds are skipped if you re-run)
python evaluate.py   # out-of-fold table: teacher / student without kNN / student with kNN
python predict.py    # test set -> two submission files (without kNN, with kNN)
```

Optional, after training (no GPU, about a minute): `python sweep.py` tries many settings of the
case-based classifier over the saved banks and prints a table. Put the settings you pick into the
`knn` block of `config.yaml` and re-run `evaluate.py` / `predict.py`.

Faster: run folds in parallel on separate GPUs (after `check.py` passed):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --fold 0 &
CUDA_VISIBLE_DEVICES=1 python train.py --fold 1 &
...
```

## Files

| File | What it is |
|---|---|
| `config.yaml` | all paths and settings |
| `data.py`, `engine.py`, `model.py` | unchanged copies of the teacher's files (data pipeline, helpers, teacher architecture) |
| `student_model.py` | architecture (ConvNeXt, metadata MLP, fusion, training head) + weight copy from the teacher |
| `student_data.py` | clinical-only dataset; training table with the teacher's folds |
| `student_engine.py` | frozen teacher, distillation loss, training loop, inference, kNN case bank, logit adjustment |
| `train.py` | 5-fold distillation; saves weights, case bank and out-of-fold predictions per fold |
| `evaluate.py` | combines the folds; the ablation (without / with kNN) next to the teacher |
| `predict.py` | 5-fold ensemble on the test set, writes the submissions |
| `sweep.py` | tries k / temperature / alpha over the saved banks; no retraining |
| `check.py` | pre-flight test of everything on a tiny subset |

## What is implemented (as decided)

- **Student inputs:** clinical image + metadata (age, sex, site, skin tone, MONET scores of the
  clinical image). `StudentNet.forward(clin, meta)` has no dermoscopic argument, and the network
  has no dermoscopic encoder or projection.
- **Initialisation:** every student layer is copied from the teacher of the same fold. Where the
  teacher's fusion takes the dermoscopic feature, the student has the teacher's learned constant
  vector. At the start the student is therefore exactly "teacher with the dermoscopic image
  hidden" (`check.py` verifies this numerically).
- **Distillation (training only):** student fold k learns from frozen teacher fold k. The teacher
  sees clinical + dermoscopic + metadata; the student sees clinical + metadata. Loss =
  class-weighted cross-entropy on the true labels (label smoothing 0.05, as the teacher)
  + logit distillation (KL, temperature 3) + feature matching (cosine distance between the
  student's and the teacher's fused 512-d feature `h`). All three use the 1/√count class weights.
- **Training:** AdamW (backbone 5e-5, rest 5e-4), cosine schedule with 1 warm-up epoch, 15 epochs,
  EMA weights, bf16, gradient clipping, same augmentation as the teacher.
- **Checkpoint:** the EMA weights after the last epoch (`checkpoint: last`), so no validation label
  influences the student. `checkpoint: val_loss` (early stopping on validation loss) is available.
- **Case bank (the classical module):** after training, the student's feature `h` and the
  confirmed diagnosis of every *training* lesion of the fold are stored (`bank.npz`). Two classical
  rules read it, and `knn.alpha` mixes them as log-probabilities:
  - **kNN** (`alpha: 0`, the default): the k = 20 most similar stored cases vote, each with weight
    exp(cosine / 0.07) after centring, times the class weight 1/√count.
  - **Nearest class mean** (`alpha: 1`): one prototype per class, the mean of that class's stored
    features. Scored by cosine / `ncm_temperature`, times the same class weight. Every class has a
    prototype, so none can be locked out by having too few cases among the k neighbours.

  Prototypes are computed from the bank only, so they add no leakage and nothing dermoscopy-derived.
  A class mean only represents its class when that class is one coherent group; for the "other"
  buckets (BEN_OTH, MAL_OTH) it averages unrelated diagnoses and means little.
- **Post-processing:** the same as the teacher: logits − 0.25 · log(training class frequency),
  then top-1. For kNN the "logits" are the log vote shares.
- **Test:** each of the 5 students predicts with its own case bank; head logits and kNN vote shares
  are averaged over the folds.

## The ablation

`evaluate.py` and `predict.py` report both decision rules from the *same* trained student:

| Row | Decision |
|---|---|
| student, head (without kNN) | the training head's class scores |
| student, kNN (final student) | the kNN vote over the case bank |

## Leakage rules that the code enforces

- Student k and teacher k are trained on the same lesions (folds ≠ k); neither sees fold k.
- The case bank of fold k contains training lesions only (`train.py` and `evaluate.py` stop if a
  validation lesion is found in it).
- Class weights, class frequencies for the logit adjustment and the bank's centring vector are
  computed from training lesions only.
- k, temperature, alpha and tau are fixed in `config.yaml`, not tuned on validation data. `sweep.py`
  tunes them on the out-of-fold set, which IS selecting on validation: report the whole sweep table,
  and fix the setting before you look at the leaderboard.
- The dermoscopic image is passed to the frozen teacher during training and to nothing else.
  Validation, the case bank and test prediction use `ClinDataset`, which never opens a
  dermoscopic file.
- Nothing is written into the teacher's run folder.

One thing the code cannot remove: each teacher checkpoint was picked at its best validation
macro-F1, so out-of-fold numbers of the teacher (and, through it, slightly of the student) are a
little optimistic. The hidden test benchmark is the clean comparison.

## Outputs (in `work_dir`)

`fold*/best.pt` (student weights), `fold*/bank.npz` (case bank), `fold*/oof.npz`, `fold*/log.csv`,
`folds.csv`, `meta_encoder.json` (copies of the teacher's), `oof_metrics.json`, `oof_per_class.csv`,
`oof_student_outputs.npz`, `submission_head_tau0.25.csv` (without kNN),
`submission_knn_tau0.25.csv` (with kNN = the student), `test_student_outputs.npz`, and
`sweep.csv` if you ran `sweep.py`.

## Notes

- Changing anything in the `knn` block does not need retraining: `sweep.py`, `evaluate.py` and
  `predict.py` rebuild the decision from the saved case banks.
- Memory: one trainable ConvNeXt-Base plus the frozen teacher needs less than teacher training;
  `check.py` prints the peak. If it runs out, set `batch_size: 8`.
