# api.py
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

MODEL_PATH = "best_model_Stacking_Linear+Tree+Cat_2025_07_20.pkl"
LOG_FILE = "prediction_history.csv"

with open(MODEL_PATH, "rb") as f:
    scaler, model = pickle.load(f)

def feature_engineering(input_data: pd.DataFrame) -> pd.DataFrame:
    df = input_data.copy()
    df["Income_Age"] = df["MedInc"] * df["HouseAge"]
    df["Rooms_per_Occup"] = df["AveRooms"] / (df["AveOccup"] + 1)
    df["Log_Pop"] = np.log(df["Population"] + 1)
    df["Near_Coast"] = (df["Longitude"] > -118).astype(int)
    return df

def predict_house_price(input_data: dict) -> float:
    df = pd.DataFrame([input_data])
    df_fe = feature_engineering(df)
    selected_features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Latitude", "Longitude",
        "Income_Age", "Rooms_per_Occup", "Near_Coast"
    ]
    df_selected = df_fe[selected_features]
    df_scaled = scaler.transform(df_selected)
    prediction = model.predict(df_scaled)[0]
    log_prediction(df_fe, prediction)
    return prediction

def log_prediction(data: pd.DataFrame, prediction: float):
    data = data.copy()
    data["MedHouseVal"] = prediction
    data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(LOG_FILE):
        data.to_csv(LOG_FILE, index=False)
    else:
        data.to_csv(LOG_FILE, mode="a", header=False, index=False)

def get_prediction_history() -> pd.DataFrame:
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame()
