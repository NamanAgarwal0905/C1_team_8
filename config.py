"""Shared configuration for the whole project (paths, constants, small helpers).

Every script imports from here, so paths only need to be changed in one place.
All scripts must be run from the repository root, e.g.  python -m cnn.train
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- data folders
DATA_DIR = ROOT / "data"
DATASET_LOCATION_FILE = DATA_DIR / "dataset_location.txt"   # written by download step
SUBSET_DIR = DATA_DIR / "subset"                            # Apple + Corn, sampled
SUBSET_TRAIN = SUBSET_DIR / "train"
SUBSET_VALID = SUBSET_DIR / "valid"
AUG_DIR = DATA_DIR / "augmented"
AUG_SVM_TRAIN = AUG_DIR / "svm" / "train"                   # originals + 3 augmentations
AUG_CNN_TRAIN = AUG_DIR / "cnn" / "train"                   # originals + 2 augmentations
FEATURE_DIR = DATA_DIR / "features"                         # cached VGG16 features
TEST_IMAGES_DIR = DATA_DIR / "test_images"                  # 10 Apple/Corn test photos

# --------------------------------------------------------------------- models
MODELS_DIR = ROOT / "models"
SVM_DIR = MODELS_DIR / "vgg_svm"
CNN_DIR = MODELS_DIR / "cnn"
SVM_MODEL_FILE = SVM_DIR / "svm_model.pkl"
SVM_SCALER_FILE = SVM_DIR / "scaler.pkl"
SVM_LABELS_FILE = SVM_DIR / "label_encoder.pkl"
CNN_MODEL_FILE = CNN_DIR / "final_plant_model.keras"
CNN_CHECKPOINT_FILE = CNN_DIR / "best_checkpoint.keras"
CNN_CLASSES_FILE = CNN_DIR / "class_names.json"
# Label order used when the CNN was trained (folders sorted alphabetically). Used automatically
# if models/cnn/class_names.json is missing, e.g. when you only copy the .keras file from Colab.
CNN_DEFAULT_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
]

RESULTS_DIR = ROOT / "results"

# ------------------------------------------------------------------ constants
SEED = 42
TARGET_PLANTS = ("Apple", "Corn")          # only these crops are used
TRAIN_PER_CLASS = 500
VALID_PER_CLASS = 100
SVM_AUGMENTATIONS = 3                      # augmented copies per training image
CNN_AUGMENTATIONS = 2
IMG_SIZE = 224
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# "imagenet" (default) downloads the pretrained VGG16 weights on first use.
# Set VGG_WEIGHTS=none only for offline smoke tests of the code.
_w = os.environ.get("VGG_WEIGHTS", "imagenet").strip().lower()
VGG_WEIGHTS = None if _w in ("none", "") else _w


def pretty_name(class_name: str) -> str:
    """'Corn_(maize)___Common_rust_' -> 'Corn (maize) - Common rust'."""
    crop, _, disease = class_name.partition("___")
    crop = crop.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()
    return f"{crop} - {disease}" if disease else crop


def is_healthy(class_name: str) -> bool:
    return "healthy" in class_name.lower()


def list_images(folder: Path):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
