
import hashlib
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT_DIR / "data" / "raw" / "azure_v1"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

CPU_FILE = RAW_DIR / "vm_cpu_readings.csv.gz"
METADATA_FILE = RAW_DIR / "vm_metadata.csv.gz"

SUBSET_FILE = PROCESSED_DIR / "vm_cpu_subset.csv"


def load_raw_data():
    cpu_data = pd.read_csv(CPU_FILE, compression="gzip")
    metadata = pd.read_csv(METADATA_FILE, compression="gzip")

    return cpu_data, metadata


def select_vms(cpu_data, num_vms=100):
    vm_ids = cpu_data["vm_id"].unique()

    ranked_vm_ids = sorted(
        vm_ids,
        key=lambda vm_id: hashlib.sha256(
            str(vm_id).encode()
        ).hexdigest()
    )

    selected_vms = ranked_vm_ids[:num_vms]

    subset = cpu_data[
        cpu_data["vm_id"].isin(selected_vms)
    ].copy()

    return subset


def load_dataset(num_vms=100, save=True):
    cpu_data, metadata = load_raw_data()

    cpu_subset = select_vms(
        cpu_data,
        num_vms
    )

    metadata = metadata.rename(columns={
        "max_cpu": "vm_metadata_max_cpu",
        "avg_cpu": "vm_metadata_avg_cpu"
    })

    dataset = cpu_subset.merge(
        metadata,
        on="vm_id",
        how="left",
        validate="many_to_one"
    )

    dataset = dataset.sort_values(
        ["vm_id", "timestamp"]
    ).reset_index(drop=True)

    if save:
        PROCESSED_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        dataset.to_csv(
            SUBSET_FILE,
            index=False
        )

        print(f"Subset saved to: {SUBSET_FILE}")

    return dataset


if __name__ == "__main__":
    data = load_dataset(num_vms=100)

    print("\nDataset shape:", data.shape)
    print("\nColumns:", data.columns.tolist())
    print("\nVM count:", data["vm_id"].nunique())
    print("\nPreview:")
    print(data.head())
