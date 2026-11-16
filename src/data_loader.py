
from pathlib import Path
import pandas as pd
import html


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "fastStorage"


def load_vm_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load one Bitbrains VM CSV file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Bitbrains files use semicolon separators
    df = pd.read_csv(file_path, sep=";")

    # Clean column names
    df.columns = [
        html.unescape(str(col))
        .replace("\xa0", " ")
        .strip()
        for col in df.columns
    ]

    # Remove unnecessary whitespace from string values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Convert columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add VM identifier from filename
    df["vm_id"] = file_path.stem

    return df


if __name__ == "__main__":

    sample_file = DATA_DIR / "2013-8" / "1250.csv"

    df = load_vm_data(sample_file)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTimestamp range:")
    print(df["Timestamp [ms]"].min())
    print(df["Timestamp [ms]"].max())
