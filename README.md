# MILK10k: Multimodal-to-Clinical Knowledge Distillation

Skin-lesion diagnosis is more accurate with **dermoscopy**, a magnified, polarised image of the
lesion. But a dermatoscope is often not available, and many lesions are only ever photographed with
an ordinary camera. This project asks: **can a model that sees only a clinical photo learn what a
dermoscopy-aware model knows?**

We train a **teacher** that sees a clinical photo, a dermoscopic image and patient metadata. We then
**distill** its knowledge into **students** that need only the clinical photo and the metadata.

Task: 11-class diagnosis on [MILK10k](https://doi.org/10.34970/648456) (AKIEC, BCC, BEN_OTH, BKL,
DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC), with 5,240 training lesions and 479 test lesions.

## The teacher: `Fusion-Teacher/`

![Fusion-Teacher architecture](assets/teacher_architecture.png)

Two ConvNeXt-Base encoders (ImageNet-22k pretrained, 384 px) read the clinical photo and the
dermoscopic image. A small MLP encodes the metadata (age, sex, site, skin tone, MONET concept
scores). **FiLM** lets the metadata rescale both image features, and a **per-dimension gate**
decides how much to trust dermoscopy versus the clinical photo. During training the dermoscopic
image is dropped 20 % of the time, so the teacher also learns to work without it. Size: 177 M
parameters.

## The students: `KNN-Student/` and `Halluc-Student/`

![Knowledge distillation overview](assets/kd_overview.png)

During training, the frozen teacher sees all three inputs, and the student sees only the clinical
photo and the metadata. The student learns from the true label, the teacher's soft predictions
(`L_KD`) and the teacher's features (`L_feat`). At test time the dermoscopic image is not needed.
Each student starts as a copy of the teacher with the dermoscopic branch removed (89 M parameters).

| Student | What is special |
|---|---|
| **KNN-Student** | The final decision is a kNN vote among the 20 most similar training lesions (a "case bank") |
| **Halluc-Student** | A small "hallucinator" predicts the teacher's dermoscopic feature from the photo, which gives the student an extra training signal |

Each folder has its own README with the full details.

## Checkpoints and data

Both are public on Hugging Face, and no account is needed.

| | Link | Contents |
|---|---|---|
| **Checkpoints** (~7 GB) | [praveenkinthali003/milk10k-distillation-checkpoints](https://huggingface.co/praveenkinthali003/milk10k-distillation-checkpoints) | `Fusion-Teacher/`, `KNN-Student/`, `Halluc-Student/`: 5 fold models (`fold0..4/best.pt`), `meta_encoder.json`, logs and predictions. KNN-Student also has `foldK/bank.npz`. |
| **Data** (~420 MB) | [praveenkinthali003/milk10k-distillation-data](https://huggingface.co/datasets/praveenkinthali003/milk10k-distillation-data) | The official MILK10k images and CSVs, plus our 5-fold split `splits/folds.csv` |

```bash
git clone https://github.com/SnehaLakhlan/isic-milk10k-multimodal-to-clinical-distillation
cd isic-milk10k-multimodal-to-clinical-distillation
pip install -r Halluc-Student/requirements.txt huggingface_hub

hf download praveenkinthali003/milk10k-distillation-data --repo-type dataset --local-dir ../milk10k-data
hf download praveenkinthali003/milk10k-distillation-checkpoints --local-dir ../ckpts
# only one model: add  --include "Halluc-Student/*"
```

## Reproducing the results

### 1. Set the paths

In each model's `config.yaml`, point the paths at your downloads. For example,
`Halluc-Student/config.yaml`:

```yaml
teacher_work_dir: ../../ckpts/Fusion-Teacher        # students only
data:
  train_meta_csv: ../../milk10k-data/milk10k/MILK10k_Training_Metadata.csv
  train_gt_csv:   ../../milk10k-data/milk10k/MILK10k_Training_GroundTruth.csv
  train_img_dir:  ../../milk10k-data/milk10k/MILK10k_Training_Input
  test_meta_csv:  ../../milk10k-data/milk10k/MILK10k_Test_Metadata.csv
  test_img_dir:   ../../milk10k-data/milk10k/MILK10k_Test_Input
work_dir: ../../ckpts/Halluc-Student                # the downloaded model folder
```

Relative paths are read from inside the model folder. Use absolute paths if you are unsure. Set
`work_dir` to `../../ckpts/Fusion-Teacher` for the teacher and `../../ckpts/KNN-Student` for the
KNN student.

### 2. Run with our checkpoints

| | Fusion-Teacher | KNN-Student | Halluc-Student |
|---|---|---|---|
| Validation numbers (no GPU needed) | `python tune_thresholds.py` | `python evaluate.py` | `python evaluate.py` |
| Test predictions (GPU recommended) | `python predict.py` | `python predict.py` | `python predict.py` |
| Final submission file | `submission.csv` | `submission_knn_tau0.25.csv` | `submission_tau0.25.csv` |

Run the commands from inside the model's folder. Outputs are written to `work_dir`.

### 3. Or retrain from scratch

Train the teacher first, because the students are built from its weights and reuse its split:

```bash
cd Fusion-Teacher && python train.py && python tune_thresholds.py   # ~4 h on one GPU
cd ../Halluc-Student && python train.py && python evaluate.py      # ~3-4 h (same for KNN-Student)
```

For training from scratch, set `work_dir` to a new folder (e.g. `runs/...`) instead of the
downloaded checkpoints.

## Results

5-fold out-of-fold predictions on the 5,240 training lesions: each lesion is scored by the fold model
that did not train on it.

| Model | Inputs at test time | Macro-F1 | AUC |
|---|---|---|---|
| Fusion-Teacher | clinical + dermoscopic + metadata | **0.578** | **0.913** |
| Teacher with dermoscopy hidden (baseline) | clinical + metadata | 0.481 | 0.863 |
| KNN-Student | clinical + metadata | 0.493 | 0.770 |
| **Halluc-Student** | clinical + metadata | **0.504** | 0.874 |

- **Dermoscopy matters.** Hiding it drops the teacher from 0.578 to 0.481 macro-F1.
- **Both students beat the clinical-only baseline.** Halluc-Student recovers 23 % of the gap to
  dermoscopy without ever seeing a dermoscopic image.

These numbers are slightly optimistic, because each teacher checkpoint was picked by its best
validation macro-F1. The hidden test set is the clean comparison.

## Bias and error analysis of the teacher

All numbers are from the teacher's 5-fold out-of-fold predictions. Macro Dice uses the official rule
(each class positive at p ≥ 0.5). Confidence intervals are 95 % lesion-level bootstrap intervals
(B = 1,000).

**Strengths**

- **Fusion helps.** Adding dermoscopy improved macro Dice by **0.104** (95 % CI 0.080–0.129) and
  accuracy from 0.722 to 0.793 (McNemar p < 10⁻⁴⁹). The gain held in 9 of 11 classes and was
  largest for AKIEC (+0.247), DF (+0.215) and VASC (+0.176). BCC false negatives fell from 592 to 261.
- **Per-class probabilities are well calibrated.** Macro one-vs-rest ECE is 0.027, which supports
  using them as soft targets for distillation. Top-label confidence is less calibrated (ECE 0.092),
  but only 0.9 % of lesions are wrong with confidence ≥ 0.9.

**Limitations**

- **Rare classes are under-recognised.** The five tail classes (202 lesions) reach macro Dice
  0.386, against 0.696 for the six common classes. BCC alone is 48 % of the data. INF sensitivity
  is 0.14, and MAL_OTH (9 lesions) is never predicted.
- **It under-calls every class at the 0.5 threshold.** For all 11 classes, it predicts fewer
  positives than there are true cases (e.g. BKL 271 vs 544), even though the ranking is strong
  (AUC 0.913 vs macro Dice 0.555). With a softmax head, many lesions have no class above 0.5.
- **Fusion does not help inflammatory lesions.** INF was the only class where the clinical-only
  branch scored higher (Dice 0.242 vs 0.212, n = 50). This is a small difference, reported as an
  observation only.

**Skin tone (exploratory)**

There is **no overall skin-tone gap after adjusting for class mix**: accuracy is 0.789 on darker
skin vs 0.787 on lighter skin. No Dice gap by age, sex, skin tone or site survived permutation
testing with FDR correction. Per-class errors do shift with skin tone, for example BCC sensitivity
is 0.71 on darker skin (tones 0–2, n = 617) vs 0.90 on lighter skin (tones 3–5, n = 4,623). This
matches how differently common each class is in the two groups (BCC is 16 % vs 53 % of lesions),
possibly because skin tone is given to the model as metadata. These per-class estimates have no
confidence intervals, so treat them as exploratory.

## Citation and licence

- **Data:** MILK study team (2025). *MILK10k* [Data set]. ISIC Archive. https://doi.org/10.34970/648456
- **Paper:** https://doi.org/10.1016/j.jid.2025.06.1594

The data and checkpoints are released under **CC-BY-NC 4.0**, for non-commercial research only.
**These models are for research only and must not be used for clinical diagnosis.**
