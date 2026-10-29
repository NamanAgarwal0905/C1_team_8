"""Step 1 - download the Kaggle 'New Plant Diseases Dataset' and remember where it is.

    python -m preprocessing.download_dataset
    python -m preprocessing.download_dataset --source "D:/path/to/new-plant-diseases-dataset"   # already downloaded

The dataset (~2 GB, 87,000+ images, 38 classes) is NOT stored in the git repo.
This script only records where it lives; step 2 then copies the Apple + Corn subset.
"""
import argparse
import sys
from pathlib import Path

import config


def find_split_dirs(root: Path):
    """Locate the train / valid class-folder trees and the folder of loose test images under root."""
    train = valid = test = None
    for p in sorted(root.rglob("*")):
        if not p.is_dir():
            continue
        has_subdirs = any(c.is_dir() for c in p.iterdir())
        if p.name == "train" and has_subdirs and train is None:
            train = p
        elif p.name == "valid" and has_subdirs and valid is None:
            valid = p
        elif p.name == "test" and test is None and any(f.suffix.lower() in config.IMAGE_EXTS for f in p.iterdir()):
            test = p      # the test folder holds loose images (test/test/*.JPG)
    return train, valid, test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="Path of an already-downloaded dataset (skips the Kaggle download)")
    args = ap.parse_args()

    if args.source:
        root = Path(args.source).expanduser().resolve()
    else:
        try:
            import kagglehub
        except ImportError:
            sys.exit("kagglehub is not installed. Run: pip install -r requirements.txt")
        print("Downloading vipoooool/new-plant-diseases-dataset from Kaggle (about 2 GB) ...")
        root = Path(kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")).resolve()

    train, valid, test = find_split_dirs(root)
    if not (train and valid):
        sys.exit(f"Could not find 'train' and 'valid' folders under {root}.\n"
                 "Check the path, or download the dataset manually from\n"
                 "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset and use --source.")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DATASET_LOCATION_FILE.write_text(str(root), encoding="utf-8")
    print(f"Dataset root : {root}\n  train: {train}\n  valid: {valid}\n  test : {test}")
    print(f"Saved location to {config.DATASET_LOCATION_FILE}\nNext: python -m preprocessing.create_subset")


if __name__ == "__main__":
    main()
