"""Step 2 - build the Apple + Corn subset used by every model.

    python -m preprocessing.create_subset                       # 500 train / 100 valid per class
    python -m preprocessing.create_subset --train 50 --valid 20 # small subset for quick tests

Output:  data/subset/train/<class>/*.jpg   data/subset/valid/<class>/*.jpg
         data/test_images/*.JPG            (the dataset's Apple/Corn test photos)
Both the VGG16+SVM and the CNN use exactly the same sampled images (seed 42),
so their accuracies are directly comparable.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import config
from preprocessing.download_dataset import find_split_dirs


def sample_split(src: Path, dst: Path, per_class: int, rng: random.Random):
    if dst.exists():
        shutil.rmtree(dst)
    classes = [d for d in sorted(src.iterdir())
               if d.is_dir() and any(plant in d.name for plant in config.TARGET_PLANTS)]
    if not classes:
        sys.exit(f"No Apple/Corn class folders found in {src}")
    for cls in classes:
        images = [f for f in sorted(cls.iterdir()) if f.suffix.lower() in config.IMAGE_EXTS]
        chosen = rng.sample(images, per_class) if len(images) > per_class else images
        out = dst / cls.name
        out.mkdir(parents=True, exist_ok=True)
        for f in chosen:
            shutil.copy2(f, out / f.name)
        print(f"  {cls.name:<55s} {len(chosen):>4d} images")
    return len(classes)


def copy_test_images(test_dir: Path):
    config.TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    if test_dir is None:
        print("No test folder found - skipping test images.")
        return
    for f in sorted(test_dir.iterdir()):
        if f.suffix.lower() in config.IMAGE_EXTS and any(p.lower() in f.name.lower() for p in config.TARGET_PLANTS):
            shutil.copy2(f, config.TEST_IMAGES_DIR / f.name)
            n += 1
    print(f"Copied {n} Apple/Corn test images to {config.TEST_IMAGES_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=config.TRAIN_PER_CLASS, help="training images per class")
    ap.add_argument("--valid", type=int, default=config.VALID_PER_CLASS, help="validation images per class")
    ap.add_argument("--source", help="dataset root (default: location saved by the download step)")
    args = ap.parse_args()

    if args.source:
        root = Path(args.source).expanduser().resolve()
    elif config.DATASET_LOCATION_FILE.exists():
        root = Path(config.DATASET_LOCATION_FILE.read_text(encoding="utf-8").strip())
    else:
        sys.exit("Dataset location unknown. Run  python -m preprocessing.download_dataset  first.")

    train, valid, test = find_split_dirs(root)
    if not (train and valid):
        sys.exit(f"train/valid folders not found under {root}")

    rng = random.Random(config.SEED)
    print(f"Sampling {args.train} training images per class from {train}")
    n = sample_split(train, config.SUBSET_TRAIN, args.train, rng)
    print(f"Sampling {args.valid} validation images per class from {valid}")
    sample_split(valid, config.SUBSET_VALID, args.valid, rng)
    copy_test_images(test)
    print(f"\nDone: {n} classes. Next: python -m preprocessing.augment")


if __name__ == "__main__":
    main()
