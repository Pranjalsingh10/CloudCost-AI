from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "fastStorage"


def load_single_csv(file_path):
    """Load a single VM CSV file."""

    df = pd.read_csv(
        file_path,
        sep=";",
        skipinitialspace=True,
        low_memory=False
    )

    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.replace("\t", "", regex=False)
        .str.strip()
    )

    # Remove empty columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Remove trailing spaces from column names
    df.columns = df.columns.str.strip()

    # VM ID comes from filename
    df["vm_id"] = file_path.stem

    return df


def get_csv_files():
    """Find CSV files recursively."""

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    files = sorted(DATA_DIR.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    return files


def load_all_data():
    """Load all CSVs and combine them."""

    files = get_csv_files()

    print(f"Found {len(files)} CSV files.")

    dataframes = []

    for i, file_path in enumerate(files, start=1):
        df = load_single_csv(file_path)

        if "Timestamp [ms]" not in df.columns:
            raise ValueError(
                f"Timestamp column missing in {file_path}"
            )

        dataframes.append(df)

        if i % 100 == 0:
            print(f"Loaded {i}/{len(files)} files")

    combined_df = pd.concat(dataframes, ignore_index=True)

    print("Combined dataset shape:", combined_df.shape)

    return combined_df


if __name__ == "__main__":
    files = get_csv_files()

    print(f"Found {len(files)} CSV files.")
    print("First file:", files[0])

    df = load_single_csv(files[0])

    print("\nSingle-file shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string())
