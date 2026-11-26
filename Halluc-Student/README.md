# MILK10k camera-only student: fine-tuned encoder + hallucinator

The student sees a clinical photo and the metadata, nothing else. Every layer starts from the
teacher and is then fine-tuned. A **hallucinator** fills the slot where the teacher's fusion
expects its dermoscopic token, and is trained against the teacher's real one.

```
clinical image ─► ConvNeXt ─► f_clin ─┬──────────────────────────────┐
                                      │                              │
metadata ─► MLP ─► m ─────────────────┼─► hallucinator ─► token ─────┤
                                      │   (K-means codebook + MLP)   │
                                      └─► FiLM + gated fusion ◄──────┘
                                                   │
                                                   ▼
                                       head ─► logit adjustment ─► diagnosis
```

**Why the hallucinator is here.** Not to supply knowledge at test time — it cannot, since it is
computed from inputs the fusion already has. Its job is to be a **training signal**: forcing the
encoder to predict the teacher's dermoscopic token makes it keep photo detail that a bare class
label would let it discard. The ablation that matters is therefore the same run with
`--halluc_weight 0`.

At initialisation the hallucinator returns the teacher's own "no dermoscopic image" constant and
its output head is zeroed, so **the student starts out as exactly the teacher run without a
dermoscopic image**. `check.py` verifies this numerically. Everything after that is improvement.

## Setup

Self-contained. Two things come from outside, both at the top of `config.yaml`:

- `teacher_work_dir` — the trained teacher's run folder (`fold0..4/best.pt`, `folds.csv`,
  `meta_encoder.json`). Only read, never written to.
- `data` — the MILK10k paths, as in the teacher's config.

The `model` block and `train.folds` / `seed` / `img_size` must stay identical to the teacher's.

## Commands

```bash
python check.py      # ~5 min. MUST say "ALL CHECKS PASSED"
python train.py      # all 5 folds (finished folds are skipped on a re-run)
python evaluate.py
python predict.py
```

Folds in parallel on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --fold 0 &
CUDA_VISIBLE_DEVICES=1 python train.py --fold 1 &
```

## The ablations

Each variant goes to its own folder, so nothing is overwritten:

```bash
python train.py --halluc_weight 0 --work_dir runs/no_halluc              # does the new loss help?
python train.py --kd_weight 0 --feat_weight 0 --halluc_weight 0 \
                --work_dir runs/no_kd                                    # the no-distillation control
python train.py --halluc_kind mlp  --work_dir runs/mlp                   # no codebook
python train.py --halluc_kind none --work_dir runs/plain                 # no hallucinator at all
python evaluate.py --runs runs/halluc,runs/no_halluc,runs/no_kd,runs/mlp,runs/plain
python predict.py  --runs runs/halluc,runs/no_kd
```

`evaluate.py` reads each run's `settings.json` and labels the rows itself, and prints the teacher
with and without dermoscopy as the two reference points plus the share of the gap each run closes.

Budget about 2.5 hours per variant. The useful set is the first three.

## The four losses

| Loss | Weight | What it does |
|---|---|---|
| cross-entropy | `ce_weight` | the true label, class-weighted 1/√count, label smoothing 0.05 |
| logit KD | `kd_weight` | match the teacher's class scores, temperature 3 |
| feature | `feat_weight` | cosine between the student's fused `h` and the teacher's |
| **hallucination** | `halluc_weight` | cosine between the predicted and the real dermoscopic token |

All four use the same per-class weights. Lesions with no dermoscopic image are skipped by the
hallucination loss.

## The hallucinator

- `kind: codebook` — K-means over the teacher's dermoscopic tokens for that fold's training
  lesions gives `n_codes` prototypes (classical vector quantisation). They are stored as a buffer
  and never trained; a small MLP predicts the coefficients over them. This is the classical
  component, and it keeps the predictor's rank low, which matters with ~4200 lesions.
- `kind: mlp` — the same MLP predicting all 512 dimensions, no dictionary. The ablation for
  whether the codebook earns its place.
- `kind: none` — no prediction; the teacher's constant vector, i.e. the plain student.

The codebook is fitted once per fold before training, from that fold's **training** lesions only.
It costs one forward pass over them (a couple of minutes), and `train.py` reports it.

## Leakage rules the code enforces

- Student k and teacher k are trained on the same lesions (folds ≠ k); neither has seen fold k.
- The codebook, the class weights and the class frequencies for the logit adjustment all come from
  that fold's training lesions only.
- The dermoscopic image is passed to the frozen teacher during training and to nothing else.
  Validation, the test set and `predict.py` use `ClinDataset`, which never opens a dermoscopic
  file, and `check.py` fails if the student's inputs are anything but clinical image + metadata.
- The student contains no dermoscopic encoder and no dermoscopic projection; `check.py` asserts it.
- `checkpoint: last` keeps the weights from the final epoch, so no validation label influences the
  student. `checkpoint: val_loss` is available if the logs show the last epochs are clearly worse.
- Nothing is written into the teacher's run folder.

One thing the code cannot remove: each teacher checkpoint was picked at its best validation
macro-F1, so these out-of-fold numbers inherit a little of that optimism. The hidden benchmark is
the clean comparison.

## Outputs (in `work_dir`)

`fold*/best.pt`, `fold*/log.csv`, `fold*/oof.npz` (logits, `h`, and the predicted token `x`),
`folds.csv`, `meta_encoder.json`, `settings.json`, `oof_metrics.json`, `oof_per_class.csv`,
`submission_tau0.25.csv`, `test_outputs.npz`.

## Notes

- Memory: one trainable ConvNeXt plus the frozen teacher. `check.py` prints the peak; if it runs
  out, set `batch_size: 8`.
- The saved `x` in `oof.npz` lets you measure how close the predicted token got to the teacher's
  without rerunning anything.
- Expect modest numbers. The teacher gains 0.094 macro-F1 from dermoscopy and most of that is
  information a phone photo does not contain; the encoder fine-tuning, not the hallucinated token,
  is what recovers any of it.
