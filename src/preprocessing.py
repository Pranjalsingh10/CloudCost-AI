
from pathlib import Path
import pandas as pd

from src.data_loader import get_csv_files, load_single_csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TIMESTAMP_COLUMN = "Timestamp [ms]"

ROLLING_COLUMNS = [
    "CPU usage [%]",
    "Memory usage [KB]",
    "Disk read throughput [KB/s]",
    "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]",
]


def clean_data(df):
    df = df.copy()

    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {TIMESTAMP_COLUMN}")

    # Timestamp values are Unix seconds
    df[TIMESTAMP_COLUMN] = pd.to_numeric(
        df[TIMESTAMP_COLUMN],
        errors="coerce"
    )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        unit="s",
        errors="coerce",
        utc=True
    )

    df = df.dropna(subset=[TIMESTAMP_COLUMN])

    # Remove duplicate records
    df = df.drop_duplicates()

    # Convert numeric resource columns but preserve vm_id
    for column in df.columns:
        if column not in [TIMESTAMP_COLUMN, "vm_id"]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove rows that have no resource measurements
    resource_columns = [
        c for c in df.columns
        if c not in [TIMESTAMP_COLUMN, "vm_id"]
    ]

    df = df.dropna(
        subset=resource_columns,
        how="all"
    )

    df = df.sort_values(
        TIMESTAMP_COLUMN
    ).reset_index(drop=True)

    return df


def create_features(df):
    df = df.copy()

    timestamp = df[TIMESTAMP_COLUMN]

    # Time features
    df["hour"] = timestamp.dt.hour
    df["day_of_week"] = timestamp.dt.dayofweek
    df["month"] = timestamp.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    # Rolling features per VM
    for column in ROLLING_COLUMNS:

        if column not in df.columns:
            continue

        df[f"{column}_rolling_mean_3"] = (
            df[column]
            .rolling(window=3, min_periods=1)
            .mean()
        )

        df[f"{column}_rolling_mean_6"] = (
            df[column]
            .rolling(window=6, min_periods=1)
            .mean()
        )

        df[f"{column}_rolling_std_6"] = (
            df[column]
            .rolling(window=6, min_periods=1)
            .std()
            .fillna(0)
        )

    return df


def run_preprocessing():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_path = PROCESSED_DIR / "cleaned_data.csv"
    features_path = PROCESSED_DIR / "features.csv"

    files = get_csv_files()

    print(f"Processing {len(files)} VM files.")

    # Remove old output files before starting
    for path in [cleaned_path, features_path]:
        if path.exists():
            path.unlink()

    total_rows = 0
    processed_files = 0

    for i, file_path in enumerate(files, start=1):

        # Load one VM at a time
        raw_df = load_single_csv(file_path)

        cleaned_df = clean_data(raw_df)
        features_df = create_features(cleaned_df)

        # Write incrementally
        cleaned_df.to_csv(
            cleaned_path,
            mode="a",
            index=False,
            header=(i == 1)
        )

        features_df.to_csv(
            features_path,
            mode="a",
            index=False,
            header=(i == 1)
        )

        total_rows += len(cleaned_df)
        processed_files += 1

        if i % 50 == 0 or i == len(files):
            print(
                f"Processed {i}/{len(files)} files | "
                f"Rows: {total_rows:,}"
            )

        del raw_df, cleaned_df, features_df

    print("\n========== PREPROCESSING COMPLETE ==========")
    print("VM files processed:", processed_files)
    print("Total rows:", f"{total_rows:,}")
    print("Cleaned data:", cleaned_path)
    print("Features:", features_path)

    return cleaned_path, features_path


if __name__ == "__main__":
    run_preprocessing()
