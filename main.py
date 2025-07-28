from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model from /models
model_bundle = joblib.load("models/redbridge_ai_v2.0.0.pkl")
rf_model = model_bundle['rf_model']
lgbm_model = model_bundle['lgbm_model']
preprocessor = model_bundle['preprocessor']
feature_columns = model_bundle['feature_columns']

app = FastAPI()

class Workload(BaseModel):
    cpu_utilization: float
    qos_score: float
    throughput: float
    hour: int = 12  # default to noon if missing

@app.post("/optimize")
def optimize(workload: Workload):
    x = [
        workload.cpu_utilization,
        workload.qos_score,
        workload.throughput,
        workload.hour,
        0,  # day_of_week
        1 if 9 <= workload.hour <= 17 else 0,  # is_business_hours
        0,  # is_weekend
        np.sin(2 * np.pi * workload.hour / 24),
        np.cos(2 * np.pi * workload.hour / 24)
    ]

    # pad if feature count is larger
    x += [0] * (len(feature_columns) - len(x))
    x = np.array(x[:len(feature_columns)]).reshape(1, -1)
    x_transformed = preprocessor.transform(x)

    rf_pred = rf_model.predict_proba(x_transformed)[0][1]
    lgbm_pred = lgbm_model.predict_proba(x_transformed)[0][1]
    confidence = (rf_pred + lgbm_pred) / 2

    return {
        "is_optimal": confidence > 0.5,
        "confidence_score": round(confidence, 4)
    }
