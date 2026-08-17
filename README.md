# An Interpretable Dual Graph Autoencoder for ASD Detection

**Detecting Autism Spectrum Disorder from resting-state fMRI — with an explanation built into every prediction.**

> A classifier whose predictive features *are* the explanation: every decision
> traces back to specific anatomical brain connections.

This repository implements a **dual, class-specific graph autoencoder (GAE)**
for Autism Spectrum Disorder (ASD) detection. Two autoencoders are trained
separately — one on ASD subjects, one on controls — and the **per-edge
difference in their reconstruction errors (ΔE)** becomes an interpretable
feature vector that both drives classification *and* localises atypical
connectivity.

Author: **Georgia Kapetadimitri** · Supervisor: **Eftychios Protopapadakis**
Department of Applied Informatics, University of Macedonia, Thessaloniki, Greece

---

## Method at a glance

```
Brain graph ─▶ GAE-ASD  &  GAE-Control ─▶ reconstructed FC (Â_ASD, Â_CTRL)
            ─▶ ΔE per edge ─▶ feature selection ─▶ classifier ─▶ ASD / Control
```

- **Graph.** Each subject → graph of **111 Harvard–Oxford ROIs**. Edge weights come
  from the **partial-correlation** FC matrix `A ∈ ℝ^{111×111}`. Node features are
  either the ROI correlation profile (`--indim 111`) or learned time-series
  embeddings (`--indim 96`, see step 1b).
- **Encoder / decoder.** Weighted GraphSAGE encoder → node embeddings; MLP decoder
  on paired endpoint embeddings reconstructs the edge weights → `Â`.
- **Margin-based specialization loss.** Each GAE reconstructs its own class well
  and the opposite class poorly (poster config: margin `m = 0.02`, weight
  `α = 1`, Gaussian edge augmentation `σ = 0.02`). These are set **inside**
  `03-main_dual_gae_*.py`, not via CLI.
- **Interpretable feature.** `ΔE = Err_ASD − Err_CTRL` per edge → a per-subject ΔE
  vector, followed by feature selection.
- **Classifier.** Feed-forward MLP by default; SVM / RF / Naive Bayes / logistic
  regression / KNN baselines are available via `--classifier`.

---

## Environment

The project was developed and run on **Google Colab with Google Drive**. All data
and scripts live under:

```
/content/drive/MyDrive/abide_dataset
```

In a Colab notebook, mount Drive and move into the project folder first:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/abide_dataset
```

### Install dependencies

```python
!pip install torch==2.5.1
!pip install torch-geometric==2.3.0
import torch
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install -q torch-sparse  -f https://data.pyg.org/whl/torch-{torch.__version__}.html
!pip install -q nilearn
!pip install -q deepdish==0.3.6
!pip install -q numpy==1.26.4

# Fix deepdish's deprecated np.object on newer numpy:
!sed -i 's/np.object/object/g' /usr/local/lib/python3.12/dist-packages/deepdish/io/hdf5io.py
```

| Package         | Version   |
|-----------------|-----------|
| torch           | 2.5.1     |
| torch-geometric | 2.3.0     |
| torch-scatter   | (PyG wheel for the installed torch) |
| torch-sparse    | (PyG wheel for the installed torch) |
| nilearn         | latest    |
| deepdish        | 0.3.6     |
| numpy           | 1.26.4    |

A CUDA GPU (Colab runtime) is recommended.

---

## Data — ABIDE

- Dataset: **ABIDE** (https://preprocessed-connectomes-project.org/abide/)
- Pipeline: **ccs** · Atlas: **ho** (Harvard–Oxford, **111 ROIs**)
- **862 subjects** — **397 ASD** (label `1`) / **465 Control** (label `0`).
  (871 valid subjects before excluding rare scan-length `T` values → 862.)
- Per-subject fMRI is stored as `.1D` time-series files of shape `[111, T]`
  (111 ROIs × scan length `T`).
- Downloaded subject files live in `abide_dataset/ABIDE_pcp/`.
- Download/preprocessing utilities are adapted from
  [BrainGNN_Pytorch](https://github.com/xxlya/BrainGNN_Pytorch/tree/main).

---

## Preprocessing (run once)

These only need to be run the first time (or when the data changes). In the
notebook they are commented out after the first run.

```bash
# 1. Download 862 subjects' fMRI (.1D [111, T]) and build connectivity +
#    partial-connectivity matrices.
python 01-fetch_data.py --pipeline ccs --atlas ho --download True

# 2. Write one .h5 per subject holding: corr matrix (111×111),
#    partial-corr matrix (111×111), and label. Partial corr defines edge weights.
python 02-process_data.py --atlas ho
```

---

## Run the model

The current entry point is **`03-main_dual_gae.py`**. Run one fold at a time
(10-fold CV, `--fold 0 … 9`):

```bash
python 03-main_dual_gae.py \
    --n_epochs 25 --classifier mlp \
    --hidden 128 --indim 111 --nroi 111 \
    --batchSize 16 --fold 0 --lr 0.001
```

Loop over all folds to reproduce a full cross-validation run:

```bash
for f in 0 1 2 3 4 5 6 7 8 9; do
  python 03-main_dual_gae.py --n_epochs 30 --classifier mlp \
      --hidden 128 --indim 111 --nroi 111 --batchSize 16 --fold $f --lr 0.001
done
```


### Reuse pretrained GAEs

Skip retraining the two autoencoders by loading saved checkpoints:

```bash
python 03-main_dual_gae.py \
    --n_epochs 30 --classifier mlp --hidden 128 --indim 111 --nroi 111 \
    --batchSize 16 --fold 0 --lr 0.001 \
    --load_pretrained_gae \
    --ctl_gae_ckpt dual_model_checkpoints/ASD_GAE_SAGE.pth \
    --asd_gae_ckpt dual_model_checkpoints/CTL_GAE_SAGE.pth
```

---

## Command-line arguments

| Flag                   | Meaning                                             | Example values            |
|------------------------|-----------------------------------------------------|---------------------------|
| `--n_epochs`           | Training epochs                                     | `20`, `25`, `45`          |
| `--classifier`         | Final classifier head                               | `mlp`, `svm`, `rf`, `nb`, `logreg`, `knn` |
| `--hidden`             | Hidden dimension                                    | `128`                     |
| `--indim`              | Node feature dim (111 = corr profile) | `111`       |
| `--nroi`               | Number of ROIs                                      | `111`                     |
| `--batchSize`          | Batch size                                          | `8`, `16`                 |
| `--fold`               | CV fold index                                       | `0 … 9`                   |
| `--lr`                 | Learning rate                                       | `0.001`                   |
| `--load_pretrained_gae`| Load saved GAE weights instead of training          | flag                      |
| `--ctl_gae_ckpt`       | Control-GAE checkpoint path                          | `dual_model_checkpoints/…` |
| `--asd_gae_ckpt`       | ASD-GAE checkpoint path                              | `dual_model_checkpoints/…` |

Preprocessing scripts additionally accept `--pipeline ccs`, `--atlas ho`, and
`--download True`.

---

## Repository layout

```
abide_dataset/
├── ABIDE_pcp/                         # downloaded ABIDE subjects (ccs pipeline)
│   └── ccs/filt_noglobal/processed/   # processed graphs (e.g. data.pt)
├── dual_model_checkpoints/            # saved GAE weights (ASD_GAE_SAGE.pth, …)
├── 01-fetch_data.py
├── 02-process_data.py
├── 03-main_dual_gae.py             # current main script
└── ASD_Dual_GAE.ipynb                 # notebook that orchestrates the above
```

---

## Interpretability outputs

- **ΔE matrix** — the unique edges as an upper-triangular heatmap
  (`ΔE > 0` vs `ΔE < 0`), showing where the two models disagree.
- **Connectogram** — flagged edges arranged by anatomical lobe.
- **Top-ranked ROIs** across folds: left inferior temporal gyrus (node 82),
  right angular gyrus (node 34), left superior parietal lobule (node 80).

---

## Citation

```bibtex
@inproceedings{kapetadimitri2026dualgae,
  title     = {An Interpretable Dual Graph Autoencoder Framework for Autism
               Spectrum Disorder Detection from Resting-State fMRI},
  author    = {Kapetadimitri, Georgia and Protopapadakis, Eftychios},
  booktitle = {<Conference>},
  year      = {2026}
}
```

## Acknowledgements

- The ABIDE consortium / Preprocessed Connectomes Project for the data.
- Download and preprocessing utilities adapted from
  [BrainGNN_Pytorch](https://github.com/xxlya/BrainGNN_Pytorch).
- Supervised by Eftychios Protopapadakis, University of Macedonia.

## Contact

Georgia Kapetadimitri — gkapet@uom.edu.gr
Department of Applied Informatics, University of Macedonia
