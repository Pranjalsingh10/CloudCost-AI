
from pathlib import Path
import gc

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

FEATURES_PATH = PROCESSED_DIR / "features.csv"
RESULTS_PATH = PROCESSED_DIR / "model_results.csv"
ANOMALIES_PATH = PROCESSED_DIR / "anomalies.csv"
MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"

CHUNK_SIZE = 100_000
MAX_TRAINING_ROWS = 200_000
SAMPLE_PER_CHUNK = 5_000


def get_feature_columns(df):
    excluded = [
        "Timestamp [ms]",
        "vm_id"
    ]

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    return [
        column for column in numeric_columns
        if column not in excluded
    ]


def collect_training_sample():
    print("Collecting training sample...")

    samples = []
    feature_columns = None

    for chunk in pd.read_csv(
        FEATURES_PATH,
        chunksize=CHUNK_SIZE
    ):

        if feature_columns is None:
            feature_columns = get_feature_columns(chunk)

        chunk = chunk[feature_columns]

        # Remove infinities
        chunk = chunk.replace(
            [float("inf"), float("-inf")],
            float("nan")
        )

        chunk = chunk.fillna(0)

        if len(chunk) > SAMPLE_PER_CHUNK:
            chunk = chunk.sample(
                n=SAMPLE_PER_CHUNK,
                random_state=42
            )

        samples.append(chunk)

    training_data = pd.concat(
        samples,
        ignore_index=True
    )

    if len(training_data) > MAX_TRAINING_ROWS:
        training_data = training_data.sample(
            n=MAX_TRAINING_ROWS,
            random_state=42
        )

    print("Training sample shape:", training_data.shape)

    del samples
    gc.collect()

    return training_data, feature_columns


def train_model(X):
    model = IsolationForest(
        n_estimators=100,
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    return model


def predict_all_data(model, feature_columns):
    print("\nPredicting anomalies in chunks...")

    # Remove old output files
    for path in [RESULTS_PATH, ANOMALIES_PATH]:
        if path.exists():
            path.unlink()

    total_rows = 0
    total_anomalies = 0
    first_chunk = True

    for chunk in pd.read_csv(
        FEATURES_PATH,
        chunksize=CHUNK_SIZE
    ):

        X = chunk[feature_columns].copy()

        X = X.replace(
            [float("inf"), float("-inf")],
            float("nan")
        ).fillna(0)

        predictions = model.predict(X)
        scores = model.decision_function(X)

        chunk["anomaly_prediction"] = predictions
        chunk["anomaly_score"] = scores
        chunk["is_anomaly"] = predictions == -1

        anomalies = chunk[
            chunk["is_anomaly"]
        ]

        chunk.to_csv(
            RESULTS_PATH,
            mode="a",
            index=False,
            header=first_chunk
        )

        anomalies.to_csv(
            ANOMALIES_PATH,
            mode="a",
            index=False,
            header=first_chunk
        )

        total_rows += len(chunk)
        total_anomalies += len(anomalies)

        first_chunk = False

        print(
            f"Processed {total_rows:,} rows | "
            f"Anomalies: {total_anomalies:,}"
        )

        del X, predictions, scores, anomalies, chunk
        gc.collect()

    return total_rows, total_anomalies


def main():
    print("\n========== CLOUDCOST TRAINING ==========")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "features.csv not found. Run preprocessing first."
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 1. Collect training sample
    X_train, feature_columns = collect_training_sample()

    if X_train.empty or not feature_columns:
        raise ValueError("No usable features available for training.")

    # 2. Train model
    print("\nTraining Isolation Forest...")

    model = train_model(X_train)

    # 3. Save model
    model_data = {
        "model": model,
        "feature_columns": feature_columns
    }

    joblib.dump(
        model_data,
        MODEL_PATH
    )

    print("Model saved:", MODEL_PATH)

    del X_train
    gc.collect()

    # 4. Predict all rows
    total_rows, total_anomalies = predict_all_data(
        model,
        feature_columns
    )

    anomaly_percentage = (
        (total_anomalies / total_rows) * 100
        if total_rows else 0
    )

    print("\n========== TRAINING SUMMARY ==========")
    print("Total rows:", f"{total_rows:,}")
    print("Detected anomalies:", f"{total_anomalies:,}")
    print("Anomaly percentage:", f"{anomaly_percentage:.2f}%")
    print("Results:", RESULTS_PATH)
    print("Anomalies:", ANOMALIES_PATH)
    print("Model:", MODEL_PATH)

    print("\nTraining completed.")


if __name__ == "__main__":
    main()
