# main.py
import joblib
from fastapi import FastAPI
import numpy as np

# Load only raw model and feature list (no custom class)
MODEL_PATH = "models/redbridge_ai_v2.0.0.pkl"
model_bundle = joblib.load(MODEL_PATH)

model = model_bundle['model']
feature_columns = model_bundle['feature_columns']

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RedBridge AI is live."}

@app.post("/predict")
def predict(payload: dict):
    try:
        # Extract input features in the correct order
        input_data = np.array([payload.get(col, 0) for col in feature_columns]).reshape(1, -1)
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
