from fastapi import FastAPI
from pydantic import BaseModel, create_model
import joblib
import numpy as np

# Load model and feature list
MODEL_PATH = "models/redbridge_ai_v2.0.0.pkl"
model, feature_columns = joblib.load(MODEL_PATH)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "RedBridge AI is live."}

# Dynamically create a Pydantic model for input validation
def create_payload_model(columns):
    fields = {col: (float, ...) for col in columns}
    return create_model("PayloadModel", **fields)

PayloadModel = create_payload_model(feature_columns)

@app.post("/predict")
def predict(payload: PayloadModel):
    try:
        input_data = np.array([getattr(payload, col) for col in feature_columns]).reshape(1, -1)
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
