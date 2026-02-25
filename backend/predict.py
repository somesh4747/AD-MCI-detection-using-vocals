"""
Model loading and prediction logic.

Loads the saved sklearn model, scaler, and feature column list once at import
time so they can be reused across requests without reloading from disk.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Optional

from feature_extraction import extract_features_from_lines, extract_features_from_file

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DIAGNOSIS_NAMES = {0: "Control", 1: "MCI"}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")

# ──────────────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ──────────────────────────────────────────────────────────────────────────────

_model = None
_scaler = None
_feature_cols = None


def _load_artifacts():
    """Load model artefacts from disk (called once on first prediction)."""
    global _model, _scaler, _feature_cols

    model_path = os.path.join(MODEL_DIR, "ad_mci_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.pkl")

    for p in (model_path, scaler_path, features_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing model artifact: {p}. "
                "Run the training notebook (Step 17) first."
            )

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    _feature_cols = joblib.load(features_path)


def get_artifacts():
    """Return (model, scaler, feature_cols), loading from disk if needed."""
    if _model is None:
        _load_artifacts()
    return _model, _scaler, _feature_cols


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict_from_cha_lines(lines: list[str], filename: str = "uploaded.cha") -> Optional[dict]:
    """
    Full pipeline: parse .cha lines → extract features → scale → predict.

    Returns a dict with prediction results, or None if feature extraction fails.
    """
    model, scaler, feature_cols = get_artifacts()

    features = extract_features_from_lines(lines)
    if features is None:
        return None

    # Build single-row DataFrame in correct column order
    X = pd.DataFrame([features])[feature_cols].astype(float)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    result = {
        "filename": filename,
        "predicted_diagnosis": DIAGNOSIS_NAMES.get(int(prediction), str(prediction)),
        "prediction_code": int(prediction),
        "confidence": round(float(probabilities[prediction]) * 100, 2),
        "probabilities": {},
        "features": features,
    }

    for idx, label in DIAGNOSIS_NAMES.items():
        if idx < len(probabilities):
            result["probabilities"][label] = round(float(probabilities[idx]) * 100, 2)

    return result


def predict_from_cha_file(file_path: str) -> Optional[dict]:
    """Convenience wrapper — reads a .cha file from disk and predicts."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return predict_from_cha_lines(lines, filename=os.path.basename(file_path))
