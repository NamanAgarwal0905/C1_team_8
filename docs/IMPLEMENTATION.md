# Implementation details

## 1. Overview
```
Kaggle dataset ─► preprocessing ─► subset (8 classes) ─► augmentation ─┬─► VGG16 + SVM ─┐
                                                                      └─► CNN ─────────┤
                                                                                       ▼
                                         photo ─► classifier ─► disease class ─► severity analysis ─► % affected + stage
                                                                              (Flask web app wraps all of this)
```
The code is split into four parts (one folder each) plus the web app. All paths and constants live in `config.py`.

## 2. Part 1 — `preprocessing/`
* **download_dataset.py** — `kagglehub.dataset_download(...)`; stores the dataset root in `data/dataset_location.txt`
  (or accepts `--source`). Finds `train/`, `valid/` (class folders) and the loose-image `test/` folder automatically.
* **create_subset.py** — keeps class folders whose name contains *Apple* or *Corn* (8 classes), samples
  500 (train) / 100 (valid) images per class with `random.Random(42)`, copies the 10 Apple/Corn test photos.
  Both models use the same sampled images, so the comparison is fair.
* **augment.py** — training images only. SVM: 3 copies per image, each with 1–3 random transforms out of 9
  (identity, h-flip, v-flip, 90° rotation, gamma 0.8–1.2, Gaussian noise σ=10, 5×5 blur, 80 % random crop, hue ±10).
  CNN: 2 copies per image (h-flip, random 0/90/180/270° rotation, brightness ×0.8–1.2).
  Result: 16,000 (SVM) and 12,000 (CNN) training images.

## 3. Part 2 — `vgg_svm/`
* **Features:** `tf.keras.applications.VGG16(weights="imagenet", include_top=False)`, frozen, + `GlobalAveragePooling2D`
  → 512-d vector. Images: BGR → resize 224×224 → RGB → `vgg16.preprocess_input`. Extracted in batches of 32 straight
  from disk (no need to hold 16,000 images in RAM) and cached in `data/features/`.
* **Classifier:** `StandardScaler` → `SVC(kernel="rbf", probability=True)`.
  Hyper-parameters by `GridSearchCV` (C ∈ {1,5,10,20}, γ ∈ {scale, auto, 0.01, 0.1}, 3-fold CV) on a random
  3,000-feature sample → best **C = 20, γ = scale** (CV accuracy 95.2 %). Final model trained on all 16,000 features.
* **Result:** 97.88 % accuracy on 800 validation images (783 correct).
* **Saved:** `svm_model.pkl`, `scaler.pkl`, `label_encoder.pkl` (joblib) + `metadata.json`.

## 4. Part 3 — `cnn/`
```
Input 224×224×3 (pixels / 255)
Conv2D 32 (3×3, ReLU) → MaxPool 2×2 → BatchNorm
Conv2D 64 (3×3, ReLU) → MaxPool 2×2 → BatchNorm
Conv2D 128 (3×3, ReLU) → GlobalAveragePooling
Dropout 0.5 → Dense 256 (ReLU) → BatchNorm → Dense 8 (softmax)         129,736 parameters (≈ 0.5 MB)
```
Adam (lr 1e-3), categorical cross-entropy, batch 32, ≤ 30 epochs, callbacks: EarlyStopping (val_loss, patience 5,
restore best weights), ModelCheckpoint (best val_accuracy), ReduceLROnPlateau (×0.2, patience 3).
The best checkpoint is evaluated and saved as `final_plant_model.keras`; `class_names.json` stores the label order
used in training so inference can never mix up classes.
**Result in our experiment:** 99.12 % accuracy / precision / recall on 800 validation images.

## 5. Part 4 — `severity/`
`analysis.analyze_severity(img_bgr, disease_class)`:
1. Resize (max side 800), Gaussian blur 5×5, convert to HSV and LAB.
2. Three masks: **K-means** (k=4, RGB; the "diseased" cluster chosen by disease type), **colour threshold**
   (rust: HSV [10–30, ≥100, ≥50]; scab: V < 100; blight: HSV [20–35, ≥100, ≥100]; otherwise adaptive threshold on LAB a/b),
   **Canny edges** (50/150, dilated).
3. A pixel is diseased if ≥ 2 of the 3 masks agree; morphological close then open (5×5).
4. Leaf mask: grey-scale Otsu → close/open → largest external contour (filled).
5. Affected % = diseased pixels inside the leaf / leaf pixels × 100 → stage:
   **0–15 % Initial**, **15–40 % Middle**, **40–100 % Severe**.
The disease hint comes from the predicted class name (*rust*, *scab*, *blight*, else general). Healthy predictions skip
the analysis. On the 10 test photos: nine Initial, one Middle (corn common rust 15.4 %), mean 3.95 %.

## 6. Web app — `app/`
Flask: `GET /` serves the page; `POST /predict` (multipart: `image`, `model` = `cnn|svm|both`) returns JSON with
class, confidence, top-3, severity numbers and the overlay as a PNG data-URL. Models are loaded lazily and cached; uploads
are limited to 12 MB and decoded with OpenCV; errors (bad file, model not trained) are returned as readable messages.

## 7. Differences from the experiment notebooks (`notebooks/`)
* Notebook cells → importable modules with command-line options; one `config.py` for all paths.
* SVM: GPU/cuML branch removed (scikit-learn CPU only); features are extracted in batches and cached;
  grid search uses `probability=False` (faster) and the final SVM is fitted with `probability=True` for confidence scores.
* CNN: the label order is saved at training time. (In the notebook, test-time labels were typed by hand in a different
  order from the training folders, which would mislabel four classes - Black rot, healthy apple, Gray leaf spot and Northern leaf blight - although the 10 test photos happen not to include them; the project avoids this.)
  A classification report and confusion matrix are now produced for the CNN too.
* Severity: the per-pixel Python voting loop is vectorised with NumPy (identical output: checked on test crops, same %
  as the notebook function) and the matplotlib plotting is replaced by a returned overlay image.
* Reproducibility: fixed seeds (42) for Python, NumPy, TensorFlow and OpenCV K-means.

## 8. Limitations / future work
Severity is not validated against expert-annotated masks; colour rules need good lighting; the 10 test photos are few and
part of the training pool was also used, so true accuracy must be measured in a real environment (a real farm). Future work:
fine-tune EfficientNet/MobileNet, learned lesion segmentation, TFLite export for a phone app.
