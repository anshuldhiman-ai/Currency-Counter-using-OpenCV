"""
fake_check.py

PLACEHOLDER fake-note screening. This is a heuristic, NOT forensic-grade
verification — it is meant to catch obviously wrong notes (photocopies,
wrong color balance, missing security thread) and give a "looks suspicious"
signal, not a legal determination of counterfeit currency.

Two checks are combined into a confidence score:
  1. Security-thread check: genuine INR notes have a thin embedded vertical
     dashed metallic thread that appears as a bright, mostly-continuous
     vertical line under normal lighting. We look for a strong vertical
     bright-line pattern.
  2. Color-consistency check: each denomination has a fairly consistent
     dominant color/hue distribution. We compare the crop's hue histogram to
     a stored reference (currently just a rough band; replace with real
     stats once you have a labeled dataset).

--- Upgrading to a real classifier ---
Once you have a real-vs-fake labeled dataset (e.g. the Roboflow set with
classes like real_10/fake_10, real_100/fake_100, ...), replace the body of
`check_note()` with inference from a small trained CNN
(e.g. MobileNetV2 fine-tuned per denomination, or one shared binary
classifier with denomination as an input feature). Keep the same function
signature so live_count.py doesn't need to change.
"""

import cv2
import numpy as np

# Rough expected hue range (OpenCV HSV, H in 0-179) per denomination.
# These are placeholders — replace with values measured from your own
# reference note photos for real accuracy.
EXPECTED_HUE_RANGE = {
    "5": (30, 80),     # green/orange
    "10": (10, 30),    # chocolate brown
    "20": (15, 35),    # greenish-yellow
    "50": (10, 25),    # fluorescent blue-ish / grey
    "100": (110, 140), # lavender/violet
    "200": (10, 30),   # bright yellow
    "500": (100, 130), # stone grey/green
}


def _security_thread_score(note_crop_bgr: np.ndarray) -> float:
    """Looks for a bright, roughly-vertical high-contrast line.
    Returns a 0-1 confidence that a thread-like feature is present."""
    gray = cv2.cvtColor(note_crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    # Sum edge pixels per column; a real thread shows as a column (or few
    # adjacent columns) with sustained high edge density top-to-bottom.
    col_sums = edges.sum(axis=0) / 255.0
    h = note_crop_bgr.shape[0]
    if h == 0:
        return 0.0
    col_ratio = col_sums / h  # fraction of column height that is an edge

    best_col_ratio = float(col_ratio.max()) if col_ratio.size else 0.0
    # A continuous thread lights up a big fraction of the column height
    return float(np.clip(best_col_ratio * 1.5, 0.0, 1.0))


def _color_consistency_score(note_crop_bgr: np.ndarray, denomination: str) -> float:
    if denomination not in EXPECTED_HUE_RANGE:
        return 0.5  # unknown denomination, neutral score

    hsv = cv2.cvtColor(note_crop_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    lo, hi = EXPECTED_HUE_RANGE[denomination]

    in_range = np.logical_and(hue >= lo, hue <= hi)
    ratio = float(in_range.mean())
    return ratio


def check_note(note_crop_bgr: np.ndarray, denomination: str):
    """
    Returns (is_likely_genuine: bool, confidence: float, details: dict)

    confidence is 0-1. Below ~0.4 -> flag as suspicious.
    This is intentionally conservative: it should reduce false "fake" alarms
    on real notes at the cost of sometimes missing a well-made fake, since
    a false fake-alarm is more disruptive for a live counting tool than a
    missed heuristic check.
    """
    if note_crop_bgr is None or note_crop_bgr.size == 0:
        return False, 0.0, {"error": "empty crop"}

    thread_score = _security_thread_score(note_crop_bgr)
    color_score = _color_consistency_score(note_crop_bgr, denomination)

    confidence = 0.5 * thread_score + 0.5 * color_score
    is_genuine = confidence >= 0.4

    return is_genuine, confidence, {
        "thread_score": round(thread_score, 3),
        "color_score": round(color_score, 3),
    }
