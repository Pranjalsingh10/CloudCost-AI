
import pandas as pd

from src.data_loader import DATA_DIR, load_vm_data

# Load the sample VM
file_path = DATA_DIR / "2013-8" / "1250.csv"
df = load_vm_data(file_path)

timestamp_col = "Timestamp [ms]"

# Interpret timestamps as Unix seconds
timestamps_seconds = pd.to_datetime(
    df[timestamp_col],
    unit="s",
    utc=True
)

# Also interpret them as milliseconds for comparison
timestamps_milliseconds = pd.to_datetime(
    df[timestamp_col],
    unit="ms",
    utc=True
)

# Calculate time differences between consecutive observations
time_differences = df[timestamp_col].diff()

print("\n--- TIMESTAMP UNIT COMPARISON ---")
print("If interpreted as seconds:")
print(timestamps_seconds.head())

print("\nIf interpreted as milliseconds:")
print(timestamps_milliseconds.head())

print("\n--- TIME INTERVAL ANALYSIS ---")
print("Most common interval:")
print(time_differences.value_counts().head())

print("\nInterval statistics:")
print(time_differences.describe())

print("\n--- TIMESTAMP QUALITY ---")
print("Duplicate timestamps:", df[timestamp_col].duplicated().sum())
print("Missing timestamps:", df[timestamp_col].isna().sum())
print("First timestamp:", timestamps_seconds.min())
print("Last timestamp:", timestamps_seconds.max())
print("Total duration:", timestamps_seconds.max() - timestamps_seconds.min())

print("\n--- RESOURCE VALUE CHECKS ---")
resource_columns = [
    "CPU usage [%]",
    "CPU usage [MHZ]",
    "Memory usage [KB]",
    "Disk read throughput [KB/s]",
    "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]",
]

print(df[resource_columns].describe().round(2))
