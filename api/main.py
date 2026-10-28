"""FastAPI inference API for the trained customer churn model."""
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability using a trained ML pipeline. Demo project; not a production decision system.",
    version="1.0.0",
)
model = None

class CustomerInput(BaseModel):
    tenure: int = Field(ge=0, le=100, description="Customer tenure in months")
    monthly_charges: float = Field(gt=0, le=10000)
    total_charges: float = Field(ge=0, description="Total charges to date")
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    tech_support: Literal["Yes", "No", "No internet service"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    paperless_billing: Literal["Yes", "No"]
    senior_citizen: int = Field(ge=0, le=1)

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Run `python -m src.train` first.")
    frame = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = "Yes" if probability >= 0.5 else "No"
    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": "High" if probability >= 0.7 else ("Medium" if probability >= 0.4 else "Low"),
        "note": "Probability is an estimate, not a guarantee."
    }
