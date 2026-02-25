"""
FastAPI backend for Control / MCI classification from .cha files.

Endpoints
---------
POST /predict          Upload a single .cha file  → prediction JSON
POST /predict/batch    Upload multiple .cha files  → list of predictions
GET  /health           Health-check / model info
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from predict import predict_from_cha_lines, get_artifacts, DIAGNOSIS_NAMES

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MCI / Control Classifier",
    description=(
        "Upload a CHAT (.cha) transcript file and get a prediction "
        "of whether the patient is **Control** or **MCI** based on "
        "speech-pause features extracted from the file."
    ),
    version="1.0.0",
)

# Allow all origins so any frontend can call this API.
# Tighten this for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Startup event — pre-load model so the first request isn't slow
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def load_model_on_startup():
    try:
        get_artifacts()
        print("Model artifacts loaded successfully.")
    except FileNotFoundError as e:
        # Don't crash the server; the /health endpoint will report the issue.
        print(f"WARNING: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Return server status and model info."""
    try:
        model, scaler, feature_cols = get_artifacts()
        return {
            "status": "ok",
            "model_type": type(model).__name__,
            "features": feature_cols,
            "classes": DIAGNOSIS_NAMES,
        }
    except FileNotFoundError as e:
        return {"status": "error", "detail": str(e)}


@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    """
    Upload a single **.cha** file and receive a diagnosis prediction.

    Returns JSON with `predicted_diagnosis`, `confidence`, per-class
    `probabilities`, and the extracted `features`.
    """
    if not file.filename.endswith(".cha"):
        raise HTTPException(status_code=400, detail="Only .cha files are accepted.")

    content = await file.read()

    try:
        lines = content.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    result = predict_from_cha_lines(lines, filename=file.filename)

    if result is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract features from the uploaded file. "
                "Make sure it is a valid .cha transcript with *PAR: and %wor: lines."
            ),
        )

    return result


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Upload **multiple .cha files** at once and get predictions for each.

    Returns a JSON list of prediction objects (same schema as `/predict`).
    Files that fail feature extraction are included with `"error"` instead of
    prediction fields.
    """
    results = []

    for file in files:
        if not file.filename.endswith(".cha"):
            results.append({"filename": file.filename, "error": "Not a .cha file — skipped."})
            continue

        content = await file.read()

        try:
            lines = content.decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            results.append({"filename": file.filename, "error": "Encoding error."})
            continue

        result = predict_from_cha_lines(lines, filename=file.filename)

        if result is None:
            results.append({"filename": file.filename, "error": "Feature extraction failed."})
        else:
            results.append(result)

    return results
