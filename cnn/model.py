"""CNN helpers shared by training, prediction and the web app."""
import json

import cv2
import numpy as np
from PIL import Image

import config


def build_model(num_classes, input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)):
    """Compact CNN (129,736 parameters for 8 classes): 3 conv blocks + GAP + dense head."""
    import tensorflow as tf
    L = tf.keras.layers
    model = tf.keras.Sequential([
        L.Input(shape=input_shape),
        L.Conv2D(32, (3, 3), activation="relu"), L.MaxPooling2D(2, 2), L.BatchNormalization(),
        L.Conv2D(64, (3, 3), activation="relu"), L.MaxPooling2D(2, 2), L.BatchNormalization(),
        L.Conv2D(128, (3, 3), activation="relu"),
        L.GlobalAveragePooling2D(),
        L.Dropout(0.5),
        L.Dense(256, activation="relu"), L.BatchNormalization(),
        L.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy",
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model


def model_exists():
    return config.CNN_MODEL_FILE.exists()


class CnnClassifier:
    """Loads the saved CNN and classifies BGR images."""
    name = "CNN"

    def __init__(self):
        if not model_exists():
            raise FileNotFoundError("CNN model not found in models/cnn/. Train it first: python -m cnn.train")
        import tensorflow as tf
        self.model = tf.keras.models.load_model(config.CNN_MODEL_FILE, compile=False)
        if config.CNN_CLASSES_FILE.exists():
            self._classes = json.loads(config.CNN_CLASSES_FILE.read_text(encoding="utf-8"))
        else:
            self._classes = list(config.CNN_DEFAULT_CLASSES)
        if self.model.output_shape[-1] != len(self._classes):
            raise ValueError("The CNN has %d outputs but %d class names are defined." % (self.model.output_shape[-1], len(self._classes)))

    @property
    def class_names(self):
        return list(self._classes)

    def predict_proba(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype="float32")[None] / 255.0
        return self.model.predict(arr, verbose=0)[0]
