# SignatureGuard: Hybrid CNN–Transformer Model for Cross-Lingual Signature Authentication

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21300792.svg)](https://doi.org/10.5281/zenodo.21300792)

Reference implementation for the paper:

> **A Transformer and Transfer-Learning Approach to Cross-Lingual Signature
> Authentication and Forgery Detection.**

This repository contains the code underlying the study. It implements the
**hybrid CNN–Transformer architecture** (the "SignatureGuard" model) and the
training/evaluation pipeline used to produce the reported results on two
signature corpora: **ASVAR** (Arabic script) and **CEDAR** (Latin/English script).

## What the model does

Each input signature image (`224 × 224 × 3`) is processed by two parallel
streams that are then fused for classification:

- **CNN stream** — an ImageNet-pretrained backbone with a frozen convolutional
  body (ResNet50, MobileNetV2, or EfficientNetB7), producing a `d_c`-dimensional
  feature vector (2048 / 1280 / 2560 respectively).
- **Transformer stream** — a frozen ViT-B/16 or DeiT-Base/16 encoder, producing
  a 768-dimensional CLS-token embedding.
- **Fusion + head** — the two vectors are concatenated and passed through
  `Linear(d_c+768, 512) → ReLU → Dropout(0.3) → Linear(512, num_classes) → Softmax`.
  Only the classification head is trained; the backbones remain frozen.

## Repository structure

```
.
├── notebooks/
│   ├── ara-hybird-deit.ipynb   # ASVAR (Arabic) — hybrid model training & evaluation
│   └── eng-hybird-deit.ipynb   # CEDAR (English) — hybrid model training & evaluation
├── requirements.txt
├── LICENSE
└── README.md
```

Each notebook builds the hybrid configurations (ResNet50, MobileNetV2, and
EfficientNetB7 backbones paired with ViT / DeiT), trains the classification
head, and reports test-set metrics (accuracy, precision, recall, F1, confusion
matrix).

## Scope note

The notebooks provided here implement the **hybrid architecture** and the
**forgery-detection** training/evaluation pipeline on both datasets. The two
additional tasks described in the paper — *biometric identification* and
*forgery-source identification* — use the **identical pipeline** with a
different label set (signer / victim identity in place of the genuine–forged
label) and the same hyperparameters. The statistical-significance analysis
(one-way ANOVA and independent-samples t-tests) was performed separately with
`scipy.stats` on the reported accuracy values.

## Environment

Developed and run under **Python 3.10** on Kaggle GPU notebooks. Install
dependencies with:

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the tested package versions.

## Data availability

The signature images are **not** included in this repository. The notebooks
expect the data in an `ImageFolder`-style layout with one subfolder per class:

```
RealvsForg/
├── <class_0>/
├── <class_1>/
└── ...
```

Update the `dataset_folder` / `train_dir` paths at the top of each notebook to
point to your local copy. Access to the datasets used in the study should be
obtained through the sources described in the paper.

## Reproducibility notes

- A fixed random seed (`42`) is used for the reported runs. Results are
  single-run estimates.
- Training uses the Adam optimizer (lr `1e-4`), batch size 32, cross-entropy
  loss, and early stopping on validation accuracy, as described in the paper.

## Citation

If you use this code, please cite the associated paper (full bibliographic
details will be updated upon publication) and the archived software release:

> Balat, M., Mamdouh, E., Awaad, R., Elhoseny, M., & Anter, A. M. (2026).
> *SignatureGuard: A Hybrid CNN–Transformer Model for Cross-Lingual Signature
> Authentication and Forgery Detection* (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.21300792

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
