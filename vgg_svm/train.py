"""Train the VGG16 (feature extractor) + SVM classifier.

    python -m vgg_svm.train                  # full run (grid search + final SVM)
    python -m vgg_svm.train --no-tune        # skip grid search, use C=10, gamma='scale'
    python -m vgg_svm.train --reuse-features # reuse cached VGG16 features (skips the slow step)
    python -m vgg_svm.train --limit 20       # only 20 images per class (quick test)

Needs: data/augmented/svm/train and data/subset/valid  (run the preprocessing steps first).
Saves: models/vgg_svm/{svm_model.pkl, scaler.pkl, label_encoder.pkl, metadata.json}
       results/vgg_svm/{classification_report.txt, confusion_matrix.png, metrics.json}
Feature extraction is the slow part (~1 h on CPU for 16,000 images, minutes on a GPU).
"""
import argparse
import warnings
import json
import sys
import time

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

import config

warnings.filterwarnings("ignore", category=FutureWarning)   # sklearn notes SVC(probability=True) changes in v1.11
from reporting import save_confusion_matrix
from vgg_svm.model import build_extractor, extract_features


def collect(folder, limit=None):
    paths, labels = [], []
    for cls in sorted(d for d in folder.iterdir() if d.is_dir()):
        files = config.list_images(cls)
        paths += files[:limit] if limit else files
        labels += [cls.name] * len(files[:limit] if limit else files)
    return paths, np.array(labels)


def features_for(paths, extractor, batch_size, desc):
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        imgs = [cv2.imread(str(p)) for p in paths[i:i + batch_size]]
        imgs = [im for im in imgs if im is not None]
        feats.append(extract_features(extractor, imgs))
    return np.vstack(feats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-tune", action="store_true", help="skip grid search (C=10, gamma='scale')")
    ap.add_argument("--reuse-features", action="store_true", help="load cached features from data/features/")
    ap.add_argument("--limit", type=int, help="max images per class (for quick tests)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    for p in (config.AUG_SVM_TRAIN, config.SUBSET_VALID):
        if not p.exists():
            sys.exit(f"{p} not found. Run the preprocessing steps first (see README).")
    np.random.seed(config.SEED)

    train_paths, y_train_txt = collect(config.AUG_SVM_TRAIN, args.limit)
    val_paths, y_val_txt = collect(config.SUBSET_VALID, args.limit)
    le = LabelEncoder().fit(y_train_txt)
    y_train, y_val = le.transform(y_train_txt), le.transform(y_val_txt)
    print(f"{len(train_paths)} training images, {len(val_paths)} validation images, {len(le.classes_)} classes")

    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    f_train, f_val = config.FEATURE_DIR / "svm_train.npy", config.FEATURE_DIR / "svm_valid.npy"
    if args.reuse_features and f_train.exists() and f_val.exists():
        X_train, X_val = np.load(f_train), np.load(f_val)
        print("Loaded cached VGG16 features.")
    else:
        extractor = build_extractor()
        t0 = time.time()
        X_train = features_for(train_paths, extractor, args.batch_size, "VGG16 features (train)")
        X_val = features_for(val_paths, extractor, args.batch_size, "VGG16 features (valid)")
        print(f"Feature extraction took {time.time() - t0:.0f} s; feature size {X_train.shape[1]}")
        np.save(f_train, X_train)
        np.save(f_val, X_val)

    scaler = StandardScaler().fit(X_train)
    Xs_train, Xs_val = scaler.transform(X_train), scaler.transform(X_val)

    C, gamma, cv_score = 10, "scale", None
    if not args.no_tune:
        n = min(3000, len(Xs_train))
        idx = np.random.choice(len(Xs_train), n, replace=False)
        grid = GridSearchCV(SVC(kernel="rbf", probability=False, random_state=config.SEED),
                            {"C": [1, 5, 10, 20], "gamma": ["scale", "auto", 0.01, 0.1]},
                            cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
        grid.fit(Xs_train[idx], y_train[idx])
        C, gamma, cv_score = grid.best_params_["C"], grid.best_params_["gamma"], float(grid.best_score_)
        print(f"Best parameters: C={C}, gamma={gamma} (3-fold CV accuracy {cv_score:.4f})")

    t0 = time.time()
    svm = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=config.SEED)
    svm.fit(Xs_train, y_train)
    pred = svm.predict(Xs_val)
    print(f"SVM trained + evaluated in {time.time() - t0:.0f} s")

    acc = accuracy_score(y_val, pred)
    report = classification_report(y_val, pred, target_names=le.classes_, digits=2, zero_division=0)
    print(f"\nValidation accuracy: {acc * 100:.2f}%\n{report}")

    config.SVM_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm, config.SVM_MODEL_FILE)
    joblib.dump(scaler, config.SVM_SCALER_FILE)
    joblib.dump(le, config.SVM_LABELS_FILE)
    meta = {"C": C, "gamma": gamma, "cv_accuracy": cv_score, "validation_accuracy": acc,
            "train_images": len(train_paths), "validation_images": len(val_paths), "classes": list(le.classes_)}
    (config.SVM_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    out = config.RESULTS_DIR / "vgg_svm"
    out.mkdir(parents=True, exist_ok=True)
    (out / "classification_report.txt").write_text(f"Validation accuracy: {acc * 100:.2f}%\n\n{report}", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"accuracy": acc}, indent=2), encoding="utf-8")
    save_confusion_matrix(confusion_matrix(y_val, pred), le.classes_, out / "confusion_matrix.png", "VGG16 + SVM")
    print(f"Models saved to {config.SVM_DIR}; reports saved to {out}")


if __name__ == "__main__":
    main()
