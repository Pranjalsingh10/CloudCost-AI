
import pandas as pd


TARGET = "avg_cpu"

FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_6",
    "rolling_max_6",
]


def validate_data(data):
    required_columns = [
        "vm_id",
        "timestamp",
        "avg_cpu",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if data[required_columns].isnull().any().any():
        raise ValueError("Missing values found in required columns")

    if data.duplicated(["vm_id", "timestamp"]).any():
        raise ValueError("Duplicate VM timestamp records found")

    data = data.sort_values(
        ["vm_id", "timestamp"]
    ).copy()

    intervals = (
        data.groupby("vm_id")["timestamp"]
        .diff()
        .dropna()
    )

    if not intervals.eq(300).all():
        raise ValueError("Unexpected timestamp intervals detected")

    return data


def create_features(data):
    data = validate_data(data)

    grouped = data.groupby("vm_id")["avg_cpu"]

    data["lag_1"] = grouped.shift(1)
    data["lag_2"] = grouped.shift(2)
    data["lag_3"] = grouped.shift(3)
    data["lag_6"] = grouped.shift(6)
    data["lag_12"] = grouped.shift(12)

    data["rolling_mean_3"] = (
        grouped.transform(
            lambda values: values.shift(1).rolling(3).mean()
        )
    )

    data["rolling_mean_6"] = (
        grouped.transform(
            lambda values: values.shift(1).rolling(6).mean()
        )
    )

    data["rolling_std_6"] = (
        grouped.transform(
            lambda values: values.shift(1).rolling(6).std()
        )
    )

    data["rolling_max_6"] = (
        grouped.transform(
            lambda values: values.shift(1).rolling(6).max()
        )
    )

    data = data.dropna(subset=FEATURES + [TARGET])

    return data


def temporal_split(data, test_ratio=0.2):
    timestamps = sorted(data["timestamp"].unique())

    split_index = int(len(timestamps) * (1 - test_ratio))

    split_timestamp = timestamps[split_index]

    train_data = data[data["timestamp"] < split_timestamp]
    test_data = data[data["timestamp"] >= split_timestamp]

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    print("\nTemporal split timestamp:", split_timestamp)
    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))

    print(
        "Train time range:",
        train_data["timestamp"].min(),
        "to",
        train_data["timestamp"].max()
    )

    print(
        "Test time range:",
        test_data["timestamp"].min(),
        "to",
        test_data["timestamp"].max()
    )

    return X_train, X_test, y_train, y_test
