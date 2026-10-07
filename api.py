import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="FastAPI backend serving the Voting Ensemble model for fraud detection.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "best_fraud_model.pkl"

def patch_sklearn_estimator(estimator):
    """Recursively patch deserialized scikit-learn estimators for cross-version compatibility."""
    if estimator is None:
        return estimator
    
    if hasattr(estimator, 'statistics_') and not hasattr(estimator, '_fill_dtype'):
        fit_dtype = getattr(estimator, '_fit_dtype', getattr(estimator.statistics_, 'dtype', np.float64))
        setattr(estimator, '_fill_dtype', fit_dtype)
        
    if hasattr(estimator, 'steps'):
        for _, step in estimator.steps:
            patch_sklearn_estimator(step)
            
    if hasattr(estimator, 'transformers_'):
        for item in estimator.transformers_:
            if len(item) >= 2:
                patch_sklearn_estimator(item[1])
                
    if hasattr(estimator, 'named_steps'):
        for _, step in estimator.named_steps.items():
            patch_sklearn_estimator(step)
            
    if hasattr(estimator, 'estimators_'):
        for est in estimator.estimators_:
            patch_sklearn_estimator(est)
            
    if hasattr(estimator, 'named_estimators_'):
        for _, est in estimator.named_estimators_.items():
            patch_sklearn_estimator(est)
            
    return estimator

try:
    try:
        import sklearn.compose._column_transformer as _ct
        from collections import UserList
        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList(UserList):
                def __init__(self, columns=(), **kwargs):
                    super().__init__(columns)
            _ct._RemainderColsList = _RemainderColsList
    except Exception:
        pass

    loaded_model = joblib.load(MODEL_PATH)
    model = patch_sklearn_estimator(loaded_model)
    print(f"✅ Successfully loaded model from: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None


class TransactionInput(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since the first transaction", example=0.0)
    V1: float = Field(0.0, description="PCA Component 1", example=-1.359807)
    V2: float = Field(0.0, description="PCA Component 2", example=-0.072781)
    V3: float = Field(0.0, description="PCA Component 3", example=2.536347)
    V4: float = Field(0.0, description="PCA Component 4", example=1.378155)
    V5: float = Field(0.0, description="PCA Component 5", example=-0.338321)
    V6: float = Field(0.0, description="PCA Component 6", example=0.462388)
    V7: float = Field(0.0, description="PCA Component 7", example=0.239599)
    V8: float = Field(0.0, description="PCA Component 8", example=0.098698)
    V9: float = Field(0.0, description="PCA Component 9", example=0.363787)
    V10: float = Field(0.0, description="PCA Component 10", example=0.090794)
    V11: float = Field(0.0, description="PCA Component 11", example=-0.551600)
    V12: float = Field(0.0, description="PCA Component 12", example=-0.617801)
    V13: float = Field(0.0, description="PCA Component 13", example=-0.991390)
    V14: float = Field(0.0, description="PCA Component 14", example=-0.311169)
    V15: float = Field(0.0, description="PCA Component 15", example=1.468177)
    V16: float = Field(0.0, description="PCA Component 16", example=-0.470401)
    V17: float = Field(0.0, description="PCA Component 17", example=0.207971)
    V18: float = Field(0.0, description="PCA Component 18", example=0.025791)
    V19: float = Field(0.0, description="PCA Component 19", example=0.403993)
    V20: float = Field(0.0, description="PCA Component 20", example=0.251412)
    V21: float = Field(0.0, description="PCA Component 21", example=-0.018307)
    V22: float = Field(0.0, description="PCA Component 22", example=0.277838)
    V23: float = Field(0.0, description="PCA Component 23", example=-0.110474)
    V24: float = Field(0.0, description="PCA Component 24", example=0.066928)
    V25: float = Field(0.0, description="PCA Component 25", example=0.128539)
    V26: float = Field(0.0, description="PCA Component 26", example=-0.189115)
    V27: float = Field(0.0, description="PCA Component 27", example=0.133558)
    V28: float = Field(0.0, description="PCA Component 28", example=-0.021053)
    Amount: float = Field(..., description="Transaction Amount in currency", example=149.62)


class PredictionResponse(BaseModel):
    prediction_class: int
    is_fraud: bool
    status: str
    fraud_probability: float
    legitimate_probability: float
    risk_level: str


@app.get("/")
def health_check():
    return {
        "service": "Credit Card Fraud Detection API",
        "status": "online" if model is not None else "model_not_found",
        "model_file": str(MODEL_PATH)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionInput, threshold: float = 0.5):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file is not loaded.")

    try:
        data_dict = transaction.dict()
        feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df = pd.DataFrame([[data_dict[col] for col in feature_order]], columns=feature_order)

        probabilities = model.predict_proba(df)[0]
        legit_prob = float(probabilities[0])
        fraud_prob = float(probabilities[1])

        is_fraud = bool(fraud_prob >= threshold)
        pred_class = 1 if is_fraud else 0

        if fraud_prob >= 0.75:
            risk_level = "CRITICAL"
        elif fraud_prob >= 0.40:
            risk_level = "HIGH"
        elif fraud_prob >= 0.15:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return PredictionResponse(
            prediction_class=pred_class,
            is_fraud=is_fraud,
            status="FRAUDULENT TRANSACTION" if is_fraud else "LEGITIMATE TRANSACTION",
            fraud_probability=round(fraud_prob, 5),
            legitimate_probability=round(legit_prob, 5),
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
def predict_batch(transactions: List[TransactionInput], threshold: float = 0.5):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file is not loaded.")

    try:
        feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        rows = [[t.dict()[col] for col in feature_order] for t in transactions]
        df = pd.DataFrame(rows, columns=feature_order)

        probabilities = model.predict_proba(df)
        results = []
        for i, probs in enumerate(probabilities):
            fraud_prob = float(probs[1])
            is_fraud = bool(fraud_prob >= threshold)
            results.append({
                "transaction_index": i,
                "is_fraud": is_fraud,
                "prediction_class": 1 if is_fraud else 0,
                "fraud_probability": round(fraud_prob, 5),
                "risk_level": "HIGH" if fraud_prob >= 0.5 else "LOW"
            })
        return {"total_transactions": len(transactions), "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
