
from src.data_loader import load_dataset


def inspect_dataset(data):
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print("\nShape:", data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nNumber of VMs:")
    print(data["vm_id"].nunique())

    print("\nRows per VM:")
    print(data.groupby("vm_id").size().describe())

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nCPU statistics:")
    print(
        data[["min_cpu", "max_cpu", "avg_cpu"]]
        .describe()
    )

    print("\nVM categories:")
    print(data["vm_category"].value_counts(dropna=False))

    print("\nMeasurements per VM:")
    counts = data.groupby("vm_id").size()

    print("Exactly 134 measurements:", (counts == 134).all())

    print("\nTimestamp intervals:")
    intervals = (
        data.sort_values(["vm_id", "timestamp"])
        .groupby("vm_id")["timestamp"]
        .diff()
        .dropna()
    )

    print(intervals.value_counts().head())

    print("\nMetadata CPU statistics:")
    print(
        data[
            ["vm_metadata_max_cpu", "vm_metadata_avg_cpu"]
        ].describe()
    )


if __name__ == "__main__":
    dataset = load_dataset(num_vms=100)
    inspect_dataset(dataset)
