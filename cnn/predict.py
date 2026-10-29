"""Classify one image with the CNN:  python -m cnn.predict path/to/leaf.jpg"""
import argparse

import cv2
import numpy as np

import config
from cnn.model import CnnClassifier


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    img = cv2.imread(ap.parse_args().image)
    if img is None:
        raise SystemExit("Could not read the image.")
    clf = CnnClassifier()
    probs = clf.predict_proba(img)
    for i in np.argsort(probs)[::-1][:3]:
        print(f"{config.pretty_name(clf.class_names[i]):<45s} {probs[i] * 100:6.2f}%")


if __name__ == "__main__":
    main()
