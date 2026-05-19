# src/serve.py
# WHAT CHANGED vs original:
# - Added Pydantic v2 input validation (was just `data: dict`)
# - /predict now returns probability + risk label + SHAP top 3 + retention actions
# - Added /health and /metrics endpoints
# - Loads pipeline.pkl (not model.pkl + model_columns.pkl separately)
# - Applies feature_engineering before prediction (Bug 3 fix)

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import sys
sys.path.append(str(Path(__file__).parent))
from feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
START_TIME = time.time()

# ── Load artifacts at startup ──────────────────────────────────
try:
    pipeline    = joblib.load(MODELS_DIR / "pipeline.pkl")
    explainer   = joblib.load(MODELS_DIR / "explainer.pkl")
    names_data  = joblib.load(MODELS_DIR / "feature_names.pkl")
    ALL_FEATURE_NAMES = names_data["all_feature_names"]
    MODEL_LOADED = True
    logger.info("Model artifacts loaded successfully")
except Exception as e:
    logger.error(f"Could not load model: {e}. Run: python src/train.py first")
    pipeline = explainer = None
    ALL_FEATURE_NAMES = []
    MODEL_LOADED = False

with open(MODELS_DIR / "latest_metrics.json") as f:
    METRICS = json.load(f) if (MODELS_DIR / "latest_metrics.json").exists() else {}


# ── Pydantic schemas ───────────────────────────────────────────
class CustomerInput(BaseModel):
    geography: str       = Field(..., examples=["France"])
    gender: str          = Field(..., examples=["Female"])
    age: int             = Field(..., ge=18, le=100)
    credit_score: int    = Field(..., ge=300, le=900)
    tenure: int          = Field(..., ge=0, le=10)
    balance: float       = Field(..., ge=0.0)
    estimated_salary: float = Field(..., ge=0.0)
    num_of_products: int = Field(..., ge=1, le=4)
    has_cr_card: int     = Field(..., ge=0, le=1)
    is_active_member: int = Field(..., ge=0, le=1)
    customer_id: Optional[str] = None

    @field_validator("geography")
    @classmethod
    def check_geography(cls, v):
        if v not in {"France", "Germany", "Spain"}:
            raise ValueError("geography must be France, Germany, or Spain")
        return v

    @field_validator("gender")
    @classmethod
    def check_gender(cls, v):
        if v not in {"Male", "Female"}:
            raise ValueError("gender must be Male or Female")
        return v

    model_config = {"json_schema_extra": {"example": {
        "geography": "Germany", "gender": "Female", "age": 42,
        "credit_score": 600, "tenure": 3, "balance": 60000.0,
        "estimated_salary": 100000.0, "num_of_products": 2,
        "has_cr_card": 1, "is_active_member": 0, "customer_id": "CUST_001"
    }}}


# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="ChurnIQ — Customer Retention Intelligence API",
    version="2.0.0",
    description="XGBoost churn prediction with SHAP explanations and retention recommendations.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Helper functions ───────────────────────────────────────────
def customer_to_df(c: CustomerInput) -> pd.DataFrame:
    """Convert input to DataFrame with correct column names for pipeline."""
    return pd.DataFrame([{
        "Geography": c.geography, "Gender": c.gender,
        "CreditScore": c.credit_score, "Age": c.age,
        "Tenure": c.tenure, "Balance": c.balance,
        "NumOfProducts": c.num_of_products, "HasCrCard": c.has_cr_card,
        "IsActiveMember": c.is_active_member, "EstimatedSalary": c.estimated_salary,
    }])


def get_shap_top3(df_input: pd.DataFrame) -> list:
    """Return top 3 SHAP features for a single prediction."""
    if explainer is None:
        return []
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(df_input)
        sv = explainer(X_transformed)
        vals = sv.values[0] if len(sv.values.shape) == 2 else sv.values[0, :, 1]
        total = float(np.sum(np.abs(vals))) + 1e-10
        features = sorted([
            {
                "feature": ALL_FEATURE_NAMES[i],
                "shap_value": round(float(v), 4),
                "direction": "increases_risk" if v > 0 else "decreases_risk",
                "impact_pct": round(abs(v) / total * 100, 1),
            }
            for i, v in enumerate(vals)
        ], key=lambda x: abs(x["shap_value"]), reverse=True)
        return features[:3]
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        return []


def get_retention_actions(c: CustomerInput, risk: str) -> list:
    """Simple rule-based retention recommendations."""
    actions = []
    if c.balance == 0:
        actions.append({"priority": 1, "action": "Offer savings account with 4.5% APY",
                        "reason": "Zero balance is strongest churn predictor", "impact": "High"})
    if c.is_active_member == 0 and c.age >= 45:
        actions.append({"priority": 2, "action": "Assign dedicated relationship manager",
                        "reason": "Inactive senior customers have 3x higher churn rate", "impact": "High"})
    if c.num_of_products == 1:
        actions.append({"priority": 3, "action": "Offer bundled product package with 20% discount",
                        "reason": "Single-product customers churn significantly more", "impact": "High"})
    if c.geography == "Germany":
        actions.append({"priority": 4, "action": "Enrol in Germany loyalty programme with cashback",
                        "reason": "Germany has highest churn rate of all geographies", "impact": "Medium"})
    if c.credit_score < 500:
        actions.append({"priority": 5, "action": "Offer credit-building programme with secured card",
                        "reason": "Low credit score signals financial stress", "impact": "Medium"})
    if c.tenure <= 1:
        actions.append({"priority": 6, "action": "Trigger 90-day new customer success journey",
                        "reason": "First-year customers have highest exit probability", "impact": "Medium"})
    if not actions:
        actions.append({"priority": 99, "action": "Standard quarterly check-in outreach",
                        "reason": "No specific high-priority risk factors detected", "impact": "Low"})
    return sorted(actions, key=lambda x: x["priority"])[:4]


def risk_label(prob: float) -> str:
    return "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW")


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "model_version": "2.0.0",
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }

@app.get("/metrics")
def metrics():
    return METRICS

@app.post("/predict")
def predict(customer: CustomerInput):
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded — run: python src/train.py")

    # Convert to DataFrame and apply feature engineering
    df = customer_to_df(customer)
    df = engineer_features(df)          # ← fixes Bug 3

    prob = float(pipeline.predict_proba(df)[0, 1])
    label = risk_label(prob)
    shap_features = get_shap_top3(df)
    actions = get_retention_actions(customer, label)

    return {
        "customer_id": customer.customer_id,
        "churn_probability": round(prob, 4),
        "risk_label": label,
        "confidence": round(max(prob, 1 - prob), 4),
        "top_risk_factors": shap_features,
        "retention_actions": actions,
        "model_version": "2.0.0",
        "prediction_id": str(uuid.uuid4()),
    }

@app.post("/predict/batch")
def predict_batch(customers: list[CustomerInput]):
    if len(customers) > 500:
        raise HTTPException(400, "Max 500 customers per batch")
    predictions = [predict(c) for c in customers]
    probs = [p["churn_probability"] for p in predictions]
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for p in predictions:
        risk_counts[p["risk_label"]] += 1
    return {
        "total_customers": len(predictions),
        **risk_counts,
        "avg_churn_probability": round(float(np.mean(probs)), 4),
        "predictions": predictions,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True,
                app_dir="src")