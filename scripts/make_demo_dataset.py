"""OPTIONAL smoke test: create a tiny FAKE dataset so you can check the whole pipeline in minutes
without downloading Kaggle data.  The images are synthetic leaves - accuracy numbers mean nothing.

    python -m scripts.make_demo_dataset
    set VGG_WEIGHTS=none        (Windows)   /   export VGG_WEIGHTS=none   (Linux/Mac)   <- skips the ImageNet download
    python -m preprocessing.create_subset --source data/demo_dataset --train 30 --valid 10
    ... then run the normal steps (augment, vgg_svm.train --no-tune, cnn.train --epochs 2, ...)
Delete data/subset, data/augmented, data/features and models/* afterwards and run the real pipeline.
"""
import random
from pathlib import Path

import cv2
import numpy as np

import config

CLASSES = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
           "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
           "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy"]
SPOT = {"scab": (30, 40, 40), "rot": (20, 20, 20), "rust": (40, 110, 200), "spot": (150, 170, 170),
        "blight": (80, 150, 180), "healthy": None}


def leaf(kind, rng):
    img = np.full((256, 256, 3), rng.randint(20, 60), np.uint8)
    cv2.ellipse(img, (128, 128), (rng.randint(80, 100), rng.randint(105, 120)), rng.randint(0, 180), 0, 360,
                (rng.randint(30, 60), rng.randint(110, 150), rng.randint(30, 60)), -1)
    colour = next((v for k, v in SPOT.items() if k in kind.lower()), None)
    if colour:
        for _ in range(rng.randint(5, 40)):
            cv2.circle(img, (rng.randint(60, 200), rng.randint(40, 215)), rng.randint(2, 8), colour, -1)
    return img


def main():
    rng = random.Random(0)
    root = Path(config.DATA_DIR) / "demo_dataset"
    for split, n in (("train", 40), ("valid", 12)):
        for c in CLASSES:
            d = root / split / c
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                cv2.imwrite(str(d / f"{i}.jpg"), leaf(c, rng))
    t = root / "test"
    t.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(["AppleScab1", "CornCommonRust1", "AppleCedarRust1"]):
        cv2.imwrite(str(t / f"{name}.JPG"), leaf(CLASSES[[0, 5, 2][i]], rng))
    print(f"Fake dataset written to {root}")


if __name__ == "__main__":
    main()
