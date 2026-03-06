"""Flask API for Employee Access predictions."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = Path("artifacts/model.joblib")
FEATURES = [
    "RESOURCE",
    "MGR_ID",
    "ROLE_ROLLUP_1",
    "ROLE_ROLLUP_2",
    "ROLE_DEPTNAME",
    "ROLE_TITLE",
    "ROLE_FAMILY_DESC",
    "ROLE_FAMILY",
    "ROLE_CODE",
]

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Expected JSON body"}), 400

    records = payload if isinstance(payload, list) else [payload]

    try:
        frame = pd.DataFrame(records)
        frame = frame[FEATURES]
    except KeyError:
        return (
            jsonify(
                {
                    "error": "Missing one or more required feature columns",
                    "required_features": FEATURES,
                }
            ),
            400,
        )

    model = _load_model()
    predictions = model.predict(frame).tolist()
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
