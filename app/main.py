from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parents[1] / "model.pkl"
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = float(bundle["threshold"])
feature_columns = bundle["feature_columns"]

app = FastAPI(title="D2C Churn Scoring API", version="1.0.0")


class CustomerFeatures(BaseModel):
    city_tier: str
    age_group: str
    acquisition_channel: str
    loyalty_tier: str
    preferred_category: str
    marketing_consent: str
    recency_days: float
    frequency_180d: float
    monetary_180d: float
    return_rate_180d: float
    avg_discount_pct_180d: float
    avg_rating_180d: float
    category_diversity_180d: float
    ticket_count_90d: float
    negative_ticket_rate_90d: float
    avg_resolution_hours_90d: float
    days_since_signup: float
    sessions_30d: float
    product_views_30d: float
    cart_adds_30d: float
    wishlist_adds_30d: float
    abandoned_carts_30d: float
    email_opens_30d: float
    campaign_clicks_30d: float
    last_visit_days_ago: float


class PredictionResponse(BaseModel):
    churn_probability: float
    predicted_class: int
    threshold: float
    risk_explanation: str


def explain(row: dict, probability: float) -> str:
    reasons = []
    if row.get("recency_days", 0) >= 90:
        reasons.append("high purchase recency")
    if row.get("sessions_30d", 0) <= 2:
        reasons.append("low recent web/app activity")
    if row.get("ticket_count_90d", 0) >= 1 or row.get("negative_ticket_rate_90d", 0) > 0:
        reasons.append("recent support friction")
    if row.get("return_rate_180d", 0) >= 0.25:
        reasons.append("elevated return rate")
    if row.get("avg_discount_pct_180d", 0) >= 0.35:
        reasons.append("discount-sensitive behavior")
    if not reasons:
        reasons.append("overall model score")
    level = "high" if probability >= threshold else "lower"
    return f"{level} churn risk driven by " + ", ".join(reasons[:3])


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["selected_model"], "threshold": threshold}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures):
    row = payload.model_dump()
    frame = pd.DataFrame([row], columns=feature_columns)
    probability = float(model.predict_proba(frame)[0, 1])
    predicted = int(probability >= threshold)
    return PredictionResponse(
        churn_probability=round(probability, 4),
        predicted_class=predicted,
        threshold=round(threshold, 4),
        risk_explanation=explain(row, probability),
    )


@app.post("/batch_predict", response_model=List[PredictionResponse])
def batch_predict(payloads: List[CustomerFeatures]):
    rows = [p.model_dump() for p in payloads]
    frame = pd.DataFrame(rows, columns=feature_columns)
    probabilities = model.predict_proba(frame)[:, 1]
    return [
        PredictionResponse(
            churn_probability=round(float(prob), 4),
            predicted_class=int(prob >= threshold),
            threshold=round(threshold, 4),
            risk_explanation=explain(row, float(prob)),
        )
        for row, prob in zip(rows, probabilities)
    ]
