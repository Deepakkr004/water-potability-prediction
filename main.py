from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained model
model_path = "model/water_potability_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Input schema for prediction
class WaterData(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
def read_root():
    return {"message": "Water Potability Prediction API"}

@app.post("/predict/")
def predict(data: WaterData):
    # Convert input to numpy array
    features = np.array([[
        data.ph, data.Hardness, data.Solids, data.Chloramines,
        data.Sulfate, data.Conductivity, data.Organic_carbon,
        data.Trihalomethanes, data.Turbidity
    ]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    potability = "Potable" if prediction == 1 else "Not Potable"
    return {"potability": potability}
