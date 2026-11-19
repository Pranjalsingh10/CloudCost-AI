
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

MODEL_URI = "models:/CloudCostAI_CPU_Forecaster@champion"

model = mlflow.pyfunc.load_model(MODEL_URI)

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_6",
    "rolling_max_6"
]


def create_input_features(cpu_history: list[float]) -> pd.DataFrame:
    if len(cpu_history) < 12:
        raise ValueError("At least 12 CPU readings are required.")

    if any(not np.isfinite(x) or x < 0 or x > 100 for x in cpu_history):
        raise ValueError("CPU readings must be between 0 and 100.")

    history = pd.Series(cpu_history, dtype=float)

    features = {
        "lag_1": history.iloc[-1],
        "lag_2": history.iloc[-2],
        "lag_3": history.iloc[-3],
        "lag_6": history.iloc[-6],
        "lag_12": history.iloc[-12],
        "rolling_mean_3": history.iloc[-3:].mean(),
        "rolling_mean_6": history.iloc[-6:].mean(),
        "rolling_std_6": history.iloc[-6:].std(),
        "rolling_max_6": history.iloc[-6:].max()
    }

    return pd.DataFrame([features])[FEATURES]


def predict_cpu(cpu_history: list[float]) -> float:
    features = create_input_features(cpu_history)
    prediction = model.predict(features)[0]

    return float(np.clip(prediction, 0, 100))
