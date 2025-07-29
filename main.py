from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and feature columns (pre-bundled)
MODEL_PATH = "models/redbridge_ai_v2.0.0.pkl"
model_bundle = joblib.load(MODEL_PATH)

model = model_bundle['model']
feature_columns = model_bundle['feature_columns']

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RedBridge AI is live."}

# Dynamically generate Pydantic model (Pydantic v1.x compatible)
def create_payload_model(cols):
    return type("PayloadModel", (BaseModel,), {
        '__annotations__': {col: float for col in cols}
    })

PayloadModel = create_payload_model(feature_columns)

@app.post("/predict")
def predict(payload: PayloadModel):
    try:
        input_data = np.array([getattr(payload, col) for col in feature_columns]).reshape(1, -1)
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
