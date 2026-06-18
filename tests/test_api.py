import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


client = TestClient(app)
sample = {
    "city_tier": "Tier 1",
    "age_group": "18-24",
    "acquisition_channel": "Instagram",
    "loyalty_tier": "Silver",
    "preferred_category": "Makeup",
    "marketing_consent": "Yes",
    "recency_days": 107,
    "frequency_180d": 1,
    "monetary_180d": 362.73,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.23,
    "avg_rating_180d": 3.0,
    "category_diversity_180d": 1,
    "ticket_count_90d": 0,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 0.0,
    "days_since_signup": 524,
    "sessions_30d": 1,
    "product_views_30d": 4,
    "cart_adds_30d": 0,
    "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 0,
    "email_opens_30d": 2,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 20
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_one_customer():
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["predicted_class"] in [0, 1]
    assert body["risk_explanation"]


def test_batch_predict():
    response = client.post("/batch_predict", json=[sample, sample])
    assert response.status_code == 200
    assert len(response.json()) == 2
