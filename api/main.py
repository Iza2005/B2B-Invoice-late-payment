"""
FastAPI service for the B2B invoice late-payment classifier.

Here, the JSON body is read as a plain dict and validated by hand in
`validate_payload()`. This keeps the request/response contract in ordinary
Python.
"""

import pathlib

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

MODEL_PATH = pathlib.Path(__file__).parent / "model.pkl"

app = FastAPI(
    title="Invoice Late-Payment Prediction API",
    description="Predicts whether a B2B invoice will be paid late.",
    version="1.0.0",
)

# Loaded once when the module is imported (when the server starts),
# not on every request.
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
NUMERIC_FEATURES = artifact["numeric_features"]
CATEGORICAL_FEATURES = artifact["categorical_features"]
FEATURES = artifact["features"]
VALID_SALES_CHANNELS = set(artifact["sales_channel_categories"])

# field -> (python types, optional min, optional max)
NUMERIC_FIELD_RULES = {
    "invoice_amount": ((int, float), 0, None),
    "payment_terms_days": ((int,), 1, 180),
    "customer_tenure_months": ((int, float), 0, None),
    "customer_credit_score": ((int, float), 300, 850),
    "previous_late_payments_count": ((int,), 0, None),
    "avg_past_delay_days": ((int, float), 0, None),
    "number_of_open_invoices": ((int,), 0, None),
    "discount_offered_pct": ((int, float), 0, 100),
    "invoice_month": ((int,), 1, 12),
    "is_recurring_customer": ((int,), 0, 1),
    "weekday_issued": ((int,), 0, 6),
}


def validate_payload(payload: dict) -> list[str]:
    """Returns a list of human-readable error messages (empty list = valid)."""
    errors = []

    if not isinstance(payload, dict):
        return ["Request body must be a JSON object."]

    for field in FEATURES:
        if field not in payload:
            errors.append(f"Missing required field: '{field}'.")

    for field, (types, low, high) in NUMERIC_FIELD_RULES.items():
        if field not in payload:
            continue
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, types):
            errors.append(f"Field '{field}' must be a number.")
            continue
        if low is not None and value < low:
            errors.append(f"Field '{field}' must be >= {low}.")
        if high is not None and value > high:
            errors.append(f"Field '{field}' must be <= {high}.")

    if "sales_channel" in payload:
        channel = payload["sales_channel"]
        if channel not in VALID_SALES_CHANNELS:
            errors.append(
                f"Field 'sales_channel' must be one of {sorted(VALID_SALES_CHANNELS)}."
            )

    return errors


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"errors": ["Request body must be valid JSON."]})

    errors = validate_payload(payload)
    if errors:
        return JSONResponse(status_code=400, content={"errors": errors})

    row = {field: payload[field] for field in FEATURES}
    X = pd.DataFrame([row])

    prediction = model.predict(X)[0]
    probability_late = float(model.predict_proba(X)[0, 1])

    return {
        "prediction": "late" if int(prediction) == 1 else "on_time",
        "probability_late": round(probability_late, 4),
        "model_metadata": {
            "model_type": artifact["model_type"],
            "trained_at": artifact["trained_at"],
            "metrics": artifact["metrics"],
        },
    }
