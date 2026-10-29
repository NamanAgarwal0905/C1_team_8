"""VGG16 feature extractor + SVM helpers shared by training, prediction and the web app."""
import json
import warnings

import cv2
import joblib
import numpy as np

import config

warnings.filterwarnings("ignore", category=FutureWarning)   # sklearn notes SVC(probability=True) changes in v1.11


def build_extractor():
    """Frozen VGG16 (ImageNet) without the classifier head + global average pooling -> 512-d vector."""
    import tensorflow as tf
    base = tf.keras.applications.VGG16(weights=config.VGG_WEIGHTS, include_top=False,
                                       input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    base.trainable = False
    return tf.keras.Sequential([base, tf.keras.layers.GlobalAveragePooling2D()])


def preprocess_bgr(images_bgr):
    """List of BGR images (any size) -> batch ready for VGG16."""
    import tensorflow as tf
    batch = np.array([cv2.resize(i, (config.IMG_SIZE, config.IMG_SIZE)) for i in images_bgr])[..., ::-1]
    return tf.keras.applications.vgg16.preprocess_input(batch.astype("float32"))


def extract_features(extractor, images_bgr):
    return extractor.predict(preprocess_bgr(images_bgr), verbose=0)


def artifacts_exist():
    return all(p.exists() for p in (config.SVM_MODEL_FILE, config.SVM_SCALER_FILE, config.SVM_LABELS_FILE))


class SvmClassifier:
    """Loads the saved SVM + scaler + label encoder and classifies BGR images."""
    name = "VGG16 + SVM"

    def __init__(self):
        if not artifacts_exist():
            raise FileNotFoundError("VGG16+SVM model not found in models/vgg_svm/. Train it first: python -m vgg_svm.train")
        self.svm = joblib.load(config.SVM_MODEL_FILE)
        self.scaler = joblib.load(config.SVM_SCALER_FILE)
        self.labels = joblib.load(config.SVM_LABELS_FILE)
        self.extractor = build_extractor()

    @property
    def class_names(self):
        return list(self.labels.classes_)

    def predict_proba(self, image_bgr):
        feats = self.scaler.transform(extract_features(self.extractor, [image_bgr]))
        return self.svm.predict_proba(feats)[0]
