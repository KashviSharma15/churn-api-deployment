"""
Task 3 — Model Deployment: Customer Churn Prediction API
Alfido Tech Internship

Run locally:  uvicorn app.main:app --reload
Docs UI:      http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ── Load model artifacts ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

try:
    model         = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler        = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
except FileNotFoundError as e:
    raise RuntimeError(
        "model.pkl / scaler.pkl not found. "
        "Run Task 1 notebook first to generate them."
    ) from e

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn. Built for Alfido Tech Internship Task 3.",
    version="1.0.0",
)

# ── Schemas ──────────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender:             int = Field(..., ge=0, le=1,  description="0 = Female, 1 = Male")
    SeniorCitizen:      int = Field(..., ge=0, le=1,  description="0 = No, 1 = Yes")
    Partner:            int = Field(..., ge=0, le=1,  description="0 = No, 1 = Yes")
    Dependents:         int = Field(..., ge=0, le=1,  description="0 = No, 1 = Yes")
    tenure:             float = Field(..., ge=0,      description="Months with company")
    PhoneService:       int = Field(..., ge=0, le=1,  description="0 = No, 1 = Yes")
    MultipleLines:      int = Field(..., ge=0, le=2,  description="0=No, 1=No phone, 2=Yes")
    InternetService:    int = Field(..., ge=0, le=2,  description="0=DSL, 1=Fiber, 2=None")
    OnlineSecurity:     int = Field(..., ge=0, le=2,  description="0=No, 1=No internet, 2=Yes")
    OnlineBackup:       int = Field(..., ge=0, le=2,  description="0=No, 1=No internet, 2=Yes")
    DeviceProtection:   int = Field(..., ge=0, le=2,  description="0=No, 1=No internet, 2=Yes")
    TechSupport:        int = Field(..., ge=0, le=2,  description="0=No, 1=No internet, 2=Yes")
    StreamingTV:        int = Field(..., ge=0, le=2,  description="0=No, 1=No internet, 2=Yes")
    StreamingMovies:    int = Field(..., ge=0, le=2,  description="0=No, 1=No internet, 2=Yes")
    Contract:           int = Field(..., ge=0, le=2,  description="0=Month-to-month, 1=One year, 2=Two year")
    PaperlessBilling:   int = Field(..., ge=0, le=1,  description="0 = No, 1 = Yes")
    PaymentMethod:      int = Field(..., ge=0, le=3,  description="0=Bank transfer, 1=Credit card, 2=E-check, 3=Mailed check")
    MonthlyCharges:     float = Field(..., ge=0,      description="Current monthly bill (USD)")
    TotalCharges:       float = Field(..., ge=0,      description="Total amount charged (USD)")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": 1, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
                "tenure": 8, "PhoneService": 1, "MultipleLines": 0,
                "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 0,
                "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0,
                "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
                "PaymentMethod": 2, "MonthlyCharges": 72.5, "TotalCharges": 580.0
            }
        }


class PredictionResponse(BaseModel):
    churn:          bool
    probability:    float = Field(..., description="Probability of churn (0–1)")
    confidence:     str   = Field(..., description="high / medium / low")
    risk_level:     str   = Field(..., description="HIGH RISK / MEDIUM RISK / LOW RISK")
    recommendation: str

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "api": "Customer Churn Prediction",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/docs"],
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok",
        "model": "RandomForestClassifier",
        "features": len(feature_names),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    try:
        # Build feature array in the same column order as training
        values = [getattr(customer, f) for f in feature_names]
        x = np.array([values])
        x_scaled = scaler.transform(x)

        prob = float(model.predict_proba(x_scaled)[0, 1])
        churn = prob >= 0.5

        # Confidence level
        margin = abs(prob - 0.5)
        if margin > 0.35:
            confidence = "high"
        elif margin > 0.15:
            confidence = "medium"
        else:
            confidence = "low"

        # Risk tier
        if prob >= 0.70:
            risk_level = "HIGH RISK"
            recommendation = "Immediate retention action recommended — offer a contract upgrade or discount."
        elif prob >= 0.45:
            risk_level = "MEDIUM RISK"
            recommendation = "Monitor this customer — consider a loyalty check-in call."
        else:
            risk_level = "LOW RISK"
            recommendation = "Customer is stable — no immediate action needed."

        return PredictionResponse(
            churn=churn,
            probability=round(prob, 4),
            confidence=confidence,
            risk_level=risk_level,
            recommendation=recommendation,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
