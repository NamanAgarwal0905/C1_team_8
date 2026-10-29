"""Classify every image in a folder, measure the affected leaf area, and write a CSV + charts.

    python -m severity.run_batch                              # data/test_images with the CNN
    python -m severity.run_batch --model svm
    python -m severity.run_batch --images path/to/folder --model cnn

Output: results/severity/severity_results.csv, results/severity/overlays/*.png,
        results/severity/summary.png
Healthy leaves are reported as "Healthy" and are not segmented.
"""
import argparse
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from severity.analysis import analyze_severity


def get_classifier(kind):
    if kind == "svm":
        from vgg_svm.model import SvmClassifier
        return SvmClassifier()
    from cnn.model import CnnClassifier
    return CnnClassifier()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(config.TEST_IMAGES_DIR))
    ap.add_argument("--model", choices=["cnn", "svm"], default="cnn")
    args = ap.parse_args()

    from pathlib import Path
    folder = Path(args.images)
    files = config.list_images(folder) if folder.exists() else []
    if not files:
        sys.exit(f"No images found in {folder}")
    clf = get_classifier(args.model)
    names = clf.class_names

    out = config.RESULTS_DIR / "severity"
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        probs = clf.predict_proba(img)
        k = int(probs.argmax())
        cls = names[k]
        row = {"image": f.name, "predicted_class": cls, "confidence_%": round(float(probs[k]) * 100, 2)}
        if config.is_healthy(cls):
            row.update(affected_percentage=0.0, severity_category="Healthy")
        else:
            res = analyze_severity(img, cls)
            if res is None:
                row.update(affected_percentage=None, severity_category="Leaf not found")
            else:
                row.update(affected_percentage=res["affected_percentage"], severity_category=res["severity_category"])
                cv2.imwrite(str(out / "overlays" / f"{f.stem}_overlay.png"), cv2.cvtColor(res["overlay_rgb"], cv2.COLOR_RGB2BGR))
        rows.append(row)
        print(f"{f.name:<28s} {config.pretty_name(cls):<45s} {row['confidence_%']:>6.1f}%  "
              f"{row['affected_percentage']}%  {row['severity_category']}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "severity_results.csv", index=False)
    valid = df.dropna(subset=["affected_percentage"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    df["severity_category"].value_counts().plot.pie(ax=ax[0], autopct="%1.0f%%", ylabel="")
    ax[0].set_title("Severity stages")
    ax[1].hist(valid["affected_percentage"], bins=10, color="#2F6B3A")
    ax[1].set_title("Affected leaf area (%)"); ax[1].set_xlabel("%"); ax[1].set_ylabel("images")
    plt.tight_layout(); plt.savefig(out / "summary.png", dpi=150); plt.close()
    print(f"\nImages: {len(df)} | mean affected area: {valid['affected_percentage'].mean():.2f}% | "
          f"max: {valid['affected_percentage'].max():.2f}%")
    print(f"Saved {out / 'severity_results.csv'}, overlays and summary.png")


if __name__ == "__main__":
    main()
