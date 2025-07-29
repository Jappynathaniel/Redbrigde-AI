from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and metadata
MODEL_PATH = "models/redbridge_ai_v2.0.0.pkl"
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle['model']
feature_columns = model_bundle['feature_columns']

# FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "RedBridge AI is live."}

# Dynamically define the input payload schema using Pydantic
def create_payload_model(cols):
    return type("PayloadModel", (BaseModel,), {
        '__annotations__': {col: float for col in cols}
    })

PayloadModel = create_payload_model(feature_columns)

@app.post("/predict")
def predict(payload: PayloadModel):
    try:
        # Prepare input in correct order
        input_data = np.array([getattr(payload, col) for col in feature_columns]).reshape(1, -1)
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
