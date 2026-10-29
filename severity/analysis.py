"""Affected-leaf-area and severity analysis (classical computer vision, no training needed).

Pipeline (same as the experiment notebooks):
 1. resize (max side 800 px) + 5x5 Gaussian blur, convert to HSV and LAB
 2. three lesion masks: K-means colour clustering (k=4), disease-specific colour threshold, Canny edges
 3. a pixel is "diseased" if at least 2 of the 3 masks agree; clean with morphological close + open
 4. leaf mask: Otsu threshold -> largest contour
 5. affected % = diseased pixels inside the leaf / leaf pixels * 100
 6. stage: 0-15 % Initial, 15-40 % Middle, 40-100 % Severe
The disease class (predicted by the classifier) tunes steps 2-3: rust -> orange/brown,
scab -> dark, blight -> yellow/brown, otherwise adaptive LAB thresholding.
"""
import cv2
import numpy as np

STAGES = [(0, 15, "Initial stage of disease"), (15, 40, "Middle stage of disease"), (40, 100.0001, "Severe damage")]
STAGE_COLORS = {"Initial stage of disease": "#D9A21B", "Middle stage of disease": "#E67E22", "Severe damage": "#C0392B"}


def emphasis_for(disease_class):
    name = (disease_class or "").lower()
    if "rust" in name:
        return "orange_brown"
    if "scab" in name:
        return "dark"
    if "blight" in name:
        return "yellow_brown"
    return "general"


def stage_for(percent):
    for lo, hi, label in STAGES:
        if lo <= percent < hi:
            return label
    return "Severe damage"


def analyze_severity(img_bgr, disease_class=None, max_dim=800):
    """Return a dict with affected_percentage, severity_category, pixel counts and overlay/mask images.

    Returns None if no leaf could be found in the image.
    """
    if img_bgr is None:
        return None
    cv2.setRNGSeed(42)                      # reproducible K-means
    emphasis = emphasis_for(disease_class)

    h0, w0 = img_bgr.shape[:2]
    scale = min(max_dim / w0, max_dim / h0)
    img = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    v = hsv[:, :, 2]
    a, b = lab[:, :, 1], lab[:, :, 2]

    # --- mask 1: K-means colour clustering -----------------------------------
    pixels = np.float32(rgb.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    c = centers.astype(float)
    if emphasis == "orange_brown":
        cluster = int(np.argmax(c[:, 0] / (c[:, 2] + 1)))
    elif emphasis == "dark":
        cluster = int(np.argmin(c.mean(axis=1)))
    elif emphasis == "yellow_brown":
        cluster = int(np.argmax(c[:, 0] * c[:, 1] / (c[:, 2] + 1)))
    else:
        cluster = int(np.argmax(np.linalg.norm(c - c.mean(axis=0), axis=1)))
    kmeans_mask = (labels.reshape(v.shape) == cluster).astype(np.uint8) * 255

    # --- mask 2: colour threshold --------------------------------------------
    if emphasis == "orange_brown":
        color_mask = cv2.inRange(hsv, np.array([10, 100, 50]), np.array([30, 255, 255]))
    elif emphasis == "dark":
        color_mask = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY_INV)[1]
    elif emphasis == "yellow_brown":
        color_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    else:
        a_t = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        b_t = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        color_mask = cv2.bitwise_or(a_t, b_t)

    # --- mask 3: edges ---------------------------------------------------------
    edges = cv2.dilate(cv2.Canny(lab, 50, 150), np.ones((3, 3), np.uint8), iterations=1)

    # --- vote (vectorised): diseased if >= 2 of 3 masks agree -------------------
    votes = (kmeans_mask > 0).astype(np.uint8) + (color_mask > 0) + (edges > 0)
    combined = (votes >= 2).astype(np.uint8) * 255
    k5 = np.ones((5, 5), np.uint8)
    disease_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k5)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, k5)

    # --- leaf segmentation -------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, leaf = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, k5)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, k5)
    contours, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    leaf_mask = np.zeros_like(leaf)
    cv2.drawContours(leaf_mask, [biggest], 0, 255, -1)

    diseased = cv2.bitwise_and(disease_mask, leaf_mask)
    leaf_px = int(cv2.countNonZero(leaf_mask))
    dis_px = int(cv2.countNonZero(diseased))
    if leaf_px == 0:
        return None
    pct = dis_px / leaf_px * 100.0

    overlay = rgb.copy()
    cv2.drawContours(overlay, [biggest], 0, (0, 255, 0), 3)
    highlight = np.zeros_like(overlay)
    highlight[:, :, 0] = diseased
    overlay = cv2.addWeighted(highlight, 0.6, overlay, 0.4, 0)

    return {
        "affected_percentage": round(pct, 2),
        "severity_category": stage_for(pct),
        "leaf_area_pixels": leaf_px,
        "disease_area_pixels": dis_px,
        "overlay_rgb": overlay,          # leaf outline in green, diseased pixels in red
        "disease_mask": diseased,
        "leaf_mask": leaf_mask,
    }
