import pickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# -------------------------------------------------
# Load Model
# -------------------------------------------------
with open("model_2.bin", "rb") as f_in:
    model = pickle.load(f_in)


# -------------------------------------------------
# Feature Engineering
# -------------------------------------------------
def identify_rush_hour(data):
    if data['workingday'] == 1:
        if data['hr'] in [7, 8, 9, 17, 18, 19]:
            return 1
    return 0


def prepare_features(data: dict):
    data = data.copy()

    # Create rush hour feature
    data['is_rush_hour'] = identify_rush_hour(data)

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Drop unused features
    features_to_drop = [
        'instant',
        'dteday',
        'atemp',
        'casual',
        'registered',
        'cnt'
    ]

    df = df.drop(columns=[c for c in features_to_drop if c in df.columns])

    return df


# -------------------------------------------------
# Input Schema
# -------------------------------------------------
class BikeRideRequest(BaseModel):
    season: int
    yr: int
    mnth: int
    hr: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    hum: float
    windspeed: float


# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(title="UrbanRide Demand Forecasting API")


@app.post("/predict")
def predict_ride_demand(request: BikeRideRequest):
    data = request.dict()

    X = prepare_features(data)

    prediction = model.predict(X)[0]

    return {
        "predicted_bike_demand": round(float(prediction), 2)
    }
