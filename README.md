# C1_Team_8 — Leaf Disease Classification and Affected-Area Analysis

**Section:** C1 &nbsp;|&nbsp; **Team:** 8 &nbsp;|&nbsp; MNNIT Allahabad, Dept. of Computer Science and Engineering

| Name | Registration No. |
|---|---|
| Naman Agarwal | 20233195 |
| Naman Katiyar | 20233196 |
| Nishant Mohan | 20233367 |
| Nitya Binoy Vettical | 20233540 |

An end-to-end system for **apple and corn leaves** (8 classes: 6 diseases + 2 healthy) that

1. **classifies** a leaf photo with two models — *VGG16 features + SVM* and a *compact CNN trained from scratch*,
2. **measures the affected leaf area (%)** with classical computer vision and maps it to a severity stage
   (Initial 0–15 %, Middle 15–40 %, Severe 40–100 %),
3. offers a small **web page** where you upload a photo and see the result.

| Model | Validation accuracy (800 images) |
|---|---|
| VGG16 + SVM (RBF, C=20) | 97.88 % |
| Compact CNN (129,736 parameters) | 99.12 % |

Full write-up: [`docs/Final_Project_Report.pdf`](docs/Final_Project_Report.pdf) · slides: [`docs/Project_Presentation.pptx`](docs/Project_Presentation.pptx) ·
design details: [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)

---------------------------------------------------------------------------------------------------

## 1. Repository layout

```
C1_Team_8/
├── README.md                  <- you are here
├── requirements.txt           <- all Python dependencies
├── config.py                  <- every path / constant in one place
├── reporting.py               <- shared confusion-matrix plot
│
├── preprocessing/             PART 1 - dataset preparation
│   ├── download_dataset.py        download Kaggle dataset (or point to a local copy)
│   ├── create_subset.py           Apple + Corn subset (500 train / 100 valid per class) + test images
│   └── augment.py                 augmentation for the SVM set (x3) and the CNN set (x2)
├── vgg_svm/                   PART 2 - VGG16 feature extractor + SVM
│   ├── model.py  train.py  predict.py
├── cnn/                       PART 3 - CNN trained from scratch
│   ├── model.py  train.py  predict.py
├── severity/                  PART 4 - affected area + severity stage
│   ├── analysis.py                the algorithm (K-means + colour + edges + Otsu)
│   └── run_batch.py               run on a folder -> CSV, overlays, charts
│
├── app/                       web front-end (Flask)
│   ├── app.py   templates/index.html
├── models/                    <- TRAINED MODELS LIVE HERE (filled by the training scripts)
│   ├── vgg_svm/                   svm_model.pkl, scaler.pkl, label_encoder.pkl
│   └── cnn/                       final_plant_model.keras, class_names.json
├── data/                      dataset folders (created by the preprocessing scripts) - see data/README.md
├── results/                   metrics, confusion matrices, severity CSV (created by the scripts)
├── scripts/make_demo_dataset.py   optional 5-minute smoke test with fake images
├── notebooks/                 the original experiment notebooks (SVM, CNN)
└── docs/                      final report (PDF + DOCX), presentation (PPTX), IMPLEMENTATION.md
```

## 2. What you need

* **Python 3.10, 3.11 or 3.12** (3.13 works with the newest TensorFlow wheels, but 3.11 is the safest choice)
* ~6 GB free disk space (dataset download ≈ 2 GB), internet for the first run
  (dataset + the 58 MB pretrained VGG16 weights, downloaded automatically)
* A GPU is **optional** but makes CNN training and VGG16 feature extraction much faster
  (CPU: feature extraction ≈ 1 h, CNN ≈ 30 epochs × a few minutes)

## 3. Setup (once)

Open a terminal **in the repository folder** (`C1_Team_8`).

**Windows (PowerShell / CMD)**
```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Always run the commands below **from the repository root**, with the virtual environment active.

## 4. Run the full pipeline — step by step

| # | Command | What it does | Output |
|---|---|---|---|
| 1 | `python -m preprocessing.download_dataset` | downloads the Kaggle dataset (≈ 2 GB) and remembers where it is | `data/dataset_location.txt` |
| 2 | `python -m preprocessing.create_subset` | samples Apple + Corn: 500 train / 100 valid images per class (8 classes) and copies the 10 Apple/Corn test photos | `data/subset/`, `data/test_images/` |
| 3 | `python -m preprocessing.augment` | builds the augmented training sets (SVM: ×3, CNN: ×2) | `data/augmented/` |
| 4 | `python -m vgg_svm.train` | VGG16 features → grid-searched RBF SVM → evaluation | `models/vgg_svm/`, `results/vgg_svm/` |
| 5 | `python -m cnn.train` | trains the CNN (≤ 30 epochs, early stopping) → evaluation | `models/cnn/`, `results/cnn/` |
| 6 | `python -m severity.run_batch` | classifies the test photos and measures affected area/stage | `results/severity/` |
| 7 | `python -m app.app` | starts the web page → open **http://127.0.0.1:5000** | — |

Steps 4 and 5 are independent — run either or both. Step 6 uses the CNN by default
(`--model svm` for the SVM). Useful options:

```bash
python -m vgg_svm.train --reuse-features   # skip the slow VGG16 step on re-runs (features are cached)
python -m vgg_svm.train --no-tune          # skip the grid search (C=10)
python -m cnn.train --epochs 5             # shorter run
python -m severity.run_batch --images my_photos --model svm
```

**Already downloaded the dataset manually?**
```bash
python -m preprocessing.download_dataset --source "D:/datasets/new-plant-diseases-dataset"
```
(the folder must contain `train`, `valid` and `test` somewhere inside it — Kaggle:
[vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)).

### Expected results (from our experiments)

* `results/vgg_svm/classification_report.txt` → accuracy ≈ **97.9 %** (best C = 20, γ = scale)
* `results/cnn/classification_report.txt` → accuracy ≈ **99 %** (exact value varies a little between runs)
* `results/severity/severity_results.csv` → on the 10 test photos 9 are *Initial stage*, corn common rust
  reaches ≈ 15 % (*Middle stage*); mean ≈ 4 %.

## 5. The web app

```bash
python -m app.app
```
Open <http://127.0.0.1:5000>, drag a leaf photo into the box, choose **CNN**, **VGG16 + SVM** or **Compare both**,
and press *Analyze leaf*. You get the predicted disease with confidence and the top-3 classes, the affected leaf-area
percentage, the severity stage, and an overlay (green = leaf outline, red = diseased area).
Healthy leaves are reported as healthy and are not segmented. A model that has not been trained yet is greyed out and the
page tells you which command to run. To open it from another device on your network: `HOST=0.0.0.0 python -m app.app`
(Windows: `set HOST=0.0.0.0`).

Single-image command line:
```bash
python -m cnn.predict path/to/leaf.jpg
python -m vgg_svm.predict path/to/leaf.jpg
```

## 6. Where the trained models go

The training scripts save everything automatically — you never copy files by hand:

```
models/vgg_svm/  svm_model.pkl  scaler.pkl  label_encoder.pkl  metadata.json
models/cnn/      final_plant_model.keras  best_checkpoint.keras  class_names.json  metadata.json
```

**Trained on Colab / Kaggle (e.g. with the notebooks) instead?** Copy only these files, then jump straight to step 7
(`python -m app.app`):

| Copy this file from Colab | Put it in | Needed for |
|---|---|---|
| `final_plant_model.keras` (CNN) | `models/cnn/` | CNN option |
| `svm_model.pkl` | `models/vgg_svm/` | VGG16 + SVM option |
| `scaler.pkl` | `models/vgg_svm/` | VGG16 + SVM option |
| `label_encoder.pkl` | `models/vgg_svm/` | VGG16 + SVM option |

Notes: `vgg_feature_extractor.keras`, `plant_model.keras` and `class_names.txt` are **not** needed (the VGG16 weights are
re-downloaded automatically on first use, ≈ 58 MB; the CNN class order is built in). The `.pkl` files are tied to the
scikit-learn version that created them, so check it in Colab (`import sklearn; print(sklearn.__version__)`) and install the
same one locally if you see an "InconsistentVersionWarning" / unpickling error: `pip install scikit-learn==<that version>`.
If you trained with this repo's scripts on Colab, also copy `class_names.json` (optional) and `results/` (optional).

## 7. Quick smoke test (optional, ~5 minutes, no download)

Checks that your installation works using synthetic images (the accuracy is meaningless):
```bash
python -m scripts.make_demo_dataset
python -m preprocessing.create_subset --source data/demo_dataset --train 30 --valid 10
python -m preprocessing.augment
python -m vgg_svm.train --no-tune --limit 30       # set VGG_WEIGHTS=none first to skip the VGG16 download
python -m cnn.train --epochs 2
python -m app.app
```
Windows: `set VGG_WEIGHTS=none` · macOS/Linux: `export VGG_WEIGHTS=none`.
**Afterwards delete** `data/demo_dataset`, `data/subset`, `data/augmented`, `data/features`, `data/test_images/*` and
the files inside `models/` and `results/`, unset `VGG_WEIGHTS`, and run the real pipeline.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: config` / `No module named 'preprocessing'` | run from the repository root and use `python -m package.module` |
| Kaggle download asks for credentials | create a Kaggle API token (kaggle.com → Settings → *Create New Token*), save `kaggle.json` to `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`) — or download manually and use `--source` |
| `data/subset ... not found` | run steps 1–3 in order |
| VGG16 weights cannot download | check internet/proxy; the file is cached in `~/.keras/models/` afterwards |
| Out-of-memory in CNN training | `python -m cnn.train --batch-size 16` |
| Web page says a model "has not been trained yet" | run step 4 and/or 5 (models must be in `models/`) |
| TensorFlow will not install | use Python 3.10–3.12 (64-bit) |

## 9. Submission checklist (GitHub)

The repository name must follow `Section_Team_Number` → **`C1_Team_8`**. Each team member creates their own repository:

```bash
git init
git add .
git commit -m "Leaf disease classification and severity analysis - C1 Team 8"
git branch -M main
git remote add origin https://github.com/<your-username>/C1_Team_8.git
git push -u origin main
```
Checklist: ✔ source code · ✔ dataset instructions + `data/test_images` + scripts (`data/README.md`) ·
✔ implementation details (`docs/IMPLEMENTATION.md`) · ✔ run instructions (this README) ·
✔ presentation (`docs/Project_Presentation.pptx`) · ✔ final report (`docs/Final_Project_Report.pdf/.docx`).
GitHub rejects files > 100 MB: the trained models are normally well below that (check `models/vgg_svm/svm_model.pkl`;
if it is > 90 MB use [Git LFS](https://git-lfs.com) or share it by link). The full dataset is **not** committed — see `data/README.md`.
