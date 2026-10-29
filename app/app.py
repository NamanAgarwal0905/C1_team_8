"""Web front-end: upload a leaf photo -> disease class, confidence, affected area and severity.

    python -m app.app            # then open http://127.0.0.1:5000

Models are loaded from models/cnn and models/vgg_svm the first time they are used.
"""
import base64
import os
import time
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import config
from severity.analysis import STAGE_COLORS, analyze_severity

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024      # 12 MB upload limit

_cache = {}
MODEL_LABELS = {"cnn": "CNN", "svm": "VGG16 + SVM"}


def model_available(key):
    if key == "cnn":
        from cnn.model import model_exists
        return model_exists()
    from vgg_svm.model import artifacts_exist
    return artifacts_exist()


def get_classifier(key):
    if key not in _cache:
        if key == "cnn":
            from cnn.model import CnnClassifier
            _cache[key] = CnnClassifier()
        else:
            from vgg_svm.model import SvmClassifier
            _cache[key] = SvmClassifier()
    return _cache[key]


def to_data_url(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@app.route("/")
def index():
    return render_template("index.html", models={k: model_available(k) for k in MODEL_LABELS}, labels=MODEL_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    start = time.perf_counter()

    def log_time(step):
        elapsed = time.perf_counter() - start
        print(f"[TIMING] {step}: {elapsed:.3f}s", flush=True)

    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify(error="Please choose an image."), 400

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    log_time("Image decoding")

    if img is None:
        return jsonify(
            error="That file could not be read as an image (use JPG or PNG)."
        ), 400

    choice = request.form.get("model", "cnn")
    keys = list(MODEL_LABELS) if choice == "both" else [choice]

    if any(k not in MODEL_LABELS for k in keys):
        return jsonify(error="Unknown model."), 400

    missing = [MODEL_LABELS[k] for k in keys if not model_available(k)]

    log_time("Model availability check")

    if missing:
        cmd = (
            "python -m cnn.train"
            if "CNN" in missing
            else "python -m vgg_svm.train"
        )

        return jsonify(
            error=f"The {', '.join(missing)} model has not been trained yet. Run: {cmd}"
        ), 400

    results = []

    for k in keys:
        clf = get_classifier(k)

        log_time(f"{MODEL_LABELS[k]} model loading")

        probs = clf.predict_proba(img)

        log_time(f"{MODEL_LABELS[k]} prediction")

        order = np.argsort(probs)[::-1][:3]
        names = clf.class_names

        results.append({
            "model": MODEL_LABELS[k],
            "class": names[order[0]],
            "label": config.pretty_name(names[order[0]]),
            "confidence": round(float(probs[order[0]]) * 100, 2),
            "healthy": config.is_healthy(names[order[0]]),
            "top3": [
                {
                    "label": config.pretty_name(names[i]),
                    "p": round(float(probs[i]) * 100, 2)
                }
                for i in order
            ],
        })

    log_time("All classification completed")

    severity = None
    primary = results[0]

    if primary["healthy"]:
        severity = {"healthy": True}

    else:
        res = analyze_severity(img, primary["class"])

        log_time("Severity analysis")

        if res is None:
            severity = {
                "error": "No leaf could be found in this image."
            }

        else:
            severity = {
                "healthy": False,
                "percent": res["affected_percentage"],
                "stage": res["severity_category"],
                "color": STAGE_COLORS[res["severity_category"]],
                "leaf_px": res["leaf_area_pixels"],
                "disease_px": res["disease_area_pixels"],
                "overlay": to_data_url(res["overlay_rgb"]),
            }

            log_time("Overlay generation")

    log_time("Total request completed")

    return jsonify(results=results, severity=severity)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
