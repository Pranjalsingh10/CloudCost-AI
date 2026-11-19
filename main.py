
from src.data_loader import load_dataset
from src.inspect_data import inspect_dataset
from src.preprocessing import create_features, temporal_split
from train import run_training


def main():
    print("=" * 60)
    print("CLOUDCOST AI - CPU FORECASTING PIPELINE")
    print("=" * 60)

    # Step 1: Load a deterministic subset
    dataset = load_dataset(num_vms=100)

    # Step 2: Inspect the dataset
    inspect_dataset(dataset)

    # Step 3: Create forecasting features
    featured_data = create_features(dataset)

    print("\nFeatured dataset shape:", featured_data.shape)

    # Step 4: Split chronologically
    X_train, X_test, y_train, y_test = temporal_split(
        featured_data,
        test_ratio=0.2
    )

    # Step 5: Train, compare, and register models
    results, registered_model = run_training(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nPipeline completed successfully!")
    print("Registered model:", registered_model.name)
    print("Model version:", registered_model.version)


if __name__ == "__main__":
    main()
