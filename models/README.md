# models/

Trained models are written here by the training scripts (nothing needs to be copied by hand).

| Folder | Created by | Files |
|---|---|---|
| `models/vgg_svm/` | `python -m vgg_svm.train` | `svm_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `metadata.json` |
| `models/cnn/` | `python -m cnn.train` | `final_plant_model.keras`, `best_checkpoint.keras`, `class_names.json`, `metadata.json` |

**Trained elsewhere (Colab/Kaggle)?** Copy just `final_plant_model.keras` -> `models/cnn/` and `svm_model.pkl`, `scaler.pkl`, `label_encoder.pkl` -> `models/vgg_svm/`.

The web app (`python -m app.app`) and the prediction scripts load the models from these folders.
If you trained on Kaggle/Colab, download the files and drop them into the matching folder.
