"""Step 3 - data augmentation (training images only; validation stays untouched).

    python -m preprocessing.augment                 # builds both augmented sets
    python -m preprocessing.augment --only cnn      # or --only svm

 * SVM set : originals + 3 random augmentations each (1-3 of 9 transforms: flips, 90-degree
             rotation, brightness, Gaussian noise, blur, random crop, hue shift)
 * CNN set : originals + 2 augmentations each (horizontal flip, 0/90/180/270 rotation, brightness)
"""
import argparse
import random
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm

import config


# ----------------------------------------------------------------- transforms
def adjust_brightness(img, gamma=1.0):
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def gaussian_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def random_crop_and_resize(img, crop_percent=0.8):
    h, w = img.shape[:2]
    ch, cw = int(h * crop_percent), int(w * crop_percent)
    y, x = random.randint(0, h - ch), random.randint(0, w - cw)
    return cv2.resize(img[y:y + ch, x:x + cw], (w, h))


def hue_shift(img, shift=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_svm(img):
    """Apply 1-3 random transforms from the 9 used in the VGG16+SVM experiment."""
    options = [
        lambda x: x,
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.flip(x, 0),
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        lambda x: adjust_brightness(x, random.uniform(0.8, 1.2)),
        lambda x: gaussian_noise(x, 10),
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
        lambda x: random_crop_and_resize(x, 0.8),
        lambda x: hue_shift(x, random.uniform(-10, 10)),
    ]
    out = img.copy()
    for fn in random.sample(options, random.randint(1, 3)):
        out = fn(out)
    return out


def augment_cnn(img):
    """Horizontal flip, 0/90/180/270 rotation and brightness scaling (as in the CNN notebook)."""
    out = img.copy()
    if random.random() > 0.5:
        out = cv2.flip(out, 1)
    if random.random() > 0.5:
        angle = random.choice([0, 90, 180, 270])
        code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(angle)
        if code is not None:
            out = cv2.rotate(out, code)
    return cv2.convertScaleAbs(out, alpha=random.uniform(0.8, 1.2), beta=0)


# ------------------------------------------------------------------- builder
def build_augmented(src, dst, n_aug, fn, desc):
    if dst.exists():
        shutil.rmtree(dst)
    for cls in tqdm(sorted(d for d in src.iterdir() if d.is_dir()), desc=desc):
        out = dst / cls.name
        out.mkdir(parents=True, exist_ok=True)
        for f in config.list_images(cls):
            shutil.copy2(f, out / f.name)
            img = cv2.imread(str(f))
            if img is None:
                continue
            for i in range(n_aug):
                cv2.imwrite(str(out / f"aug_{i}_{f.stem}.jpg"), fn(img))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["svm", "cnn"], help="build just one of the two sets")
    args = ap.parse_args()
    if not config.SUBSET_TRAIN.exists():
        sys.exit("data/subset/train not found. Run  python -m preprocessing.create_subset  first.")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    if args.only in (None, "svm"):
        build_augmented(config.SUBSET_TRAIN, config.AUG_SVM_TRAIN, config.SVM_AUGMENTATIONS, augment_svm, "SVM augmentation")
    if args.only in (None, "cnn"):
        build_augmented(config.SUBSET_TRAIN, config.AUG_CNN_TRAIN, config.CNN_AUGMENTATIONS, augment_cnn, "CNN augmentation")
    print("Done. Next: python -m vgg_svm.train   and   python -m cnn.train")


if __name__ == "__main__":
    main()
