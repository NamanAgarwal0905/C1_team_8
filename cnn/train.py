"""Train the compact CNN from scratch.

    python -m cnn.train                 # 30 epochs max, early stopping
    python -m cnn.train --epochs 3      # quick test

Needs: data/augmented/cnn/train and data/subset/valid  (run the preprocessing steps first).
Saves: models/cnn/{final_plant_model.keras, best_checkpoint.keras, class_names.json, metadata.json}
       results/cnn/{training_history.png, history.json, classification_report.txt,
                    confusion_matrix.png, metrics.json}
A GPU is optional but makes training much faster (about 20 s per epoch on a Colab/Kaggle GPU).
"""
import argparse
import json
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import config
from cnn.model import build_model
from reporting import save_confusion_matrix


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import tensorflow as tf
    for p in (config.AUG_CNN_TRAIN, config.SUBSET_VALID):
        if not p.exists():
            sys.exit(f"{p} not found. Run the preprocessing steps first (see README).")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)
    print("GPU devices:", tf.config.list_physical_devices("GPU") or "none (training on CPU)")

    def pipeline(folder, shuffle):
        ds = tf.keras.utils.image_dataset_from_directory(
            str(folder), image_size=(config.IMG_SIZE, config.IMG_SIZE), batch_size=args.batch_size,
            label_mode="categorical", shuffle=shuffle, seed=config.SEED)
        names = ds.class_names
        return ds.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE), names

    train_ds, class_names = pipeline(config.AUG_CNN_TRAIN, True)
    val_ds, val_names = pipeline(config.SUBSET_VALID, False)
    if class_names != val_names:
        sys.exit("Class folders differ between train and valid sets.")
    print("Classes:", class_names)

    model = build_model(len(class_names))
    model.summary()

    config.CNN_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(config.CNN_CHECKPOINT_FILE), save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)

    # evaluate the best checkpoint (highest validation accuracy), as in the experiment notebook
    model = tf.keras.models.load_model(config.CNN_CHECKPOINT_FILE)
    loss, acc, prec, rec = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy : {acc:.2%}\nValidation precision: {prec:.2%}\nValidation recall   : {rec:.2%}")
    model.save(config.CNN_MODEL_FILE)
    config.CNN_CLASSES_FILE.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    # per-class report + confusion matrix
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in val_ds])
    y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=2, zero_division=0)
    print(report)

    out = config.RESULTS_DIR / "cnn"
    out.mkdir(parents=True, exist_ok=True)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (out / "history.json").write_text(json.dumps(hist), encoding="utf-8")
    (out / "classification_report.txt").write_text(
        f"Validation accuracy: {acc:.2%}  precision: {prec:.2%}  recall: {rec:.2%}\n\n{report}", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"accuracy": acc, "precision": prec, "recall": rec, "loss": loss}, indent=2), encoding="utf-8")
    (config.CNN_DIR / "metadata.json").write_text(json.dumps(
        {"validation_accuracy": acc, "epochs_run": len(hist["loss"]), "parameters": int(model.count_params()),
         "classes": class_names}, indent=2), encoding="utf-8")
    save_confusion_matrix(confusion_matrix(y_true, y_pred), class_names, out / "confusion_matrix.png", "CNN")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["accuracy"], label="train"); ax[0].plot(hist["val_accuracy"], label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["loss"], label="train"); ax[1].plot(hist["val_loss"], label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out / "training_history.png", dpi=150); plt.close()
    print(f"Model saved to {config.CNN_MODEL_FILE}; reports saved to {out}")


if __name__ == "__main__":
    main()
