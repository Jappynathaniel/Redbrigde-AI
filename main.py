from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# Load the trained model and components
MODEL_PATH = "models/redbridge_ai_v2.0.0.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ Model file not found. Make sure redbridge_ai_v2.0.0.pkl exists.")

model_bundle = joblib.load(MODEL_PATH)
trainer = model_bundle['trainer']
api = model_bundle['validator']  # Or `OptimizationAPI` if saved in bundle
config = model_bundle['config']

# Setup FastAPI
app = FastAPI(title="RedBridge AI API", version="2.0.0")

# Request model
class WorkloadRequest(BaseModel):
    id: str
    cpu_utilization: float
    qos_score: float
    throughput: float
    hour: int

@app.post("/optimize")
def optimize_workload(request: WorkloadRequest):
    if not trainer or not trainer.model:
        raise HTTPException(status_code=500, detail="Model not ready")

    # Simplified feature vector from the request
    features = [request.cpu_utilization, request.qos_score, request.throughput, request.hour]
    while len(features) < len(trainer.feature_columns):
        features.append(0)

    X = trainer.preprocessor.transform([features])
    rf_model, lgbm_model = trainer.model

    rf_conf = rf_model.predict_proba(X)[0][1]
    lgbm_conf = lgbm_model.predict_proba(X)[0][1]
    ensemble_conf = (rf_conf + lgbm_conf) / 2

    return {
        "workload_id": request.id,
        "confidence": ensemble_conf,
        "is_optimal": ensemble_conf > 0.5
    }
