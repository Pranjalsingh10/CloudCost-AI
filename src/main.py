
from src.preprocessing import run_preprocessing
from src.train import main as run_training


def main():
    print("=" * 60)
    print("CLOUDCOST - PHASE 1 PIPELINE")
    print("=" * 60)

    print("\n[1/2] Starting preprocessing...")
    run_preprocessing()

    print("\n[2/2] Starting model training...")
    run_training()

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
  