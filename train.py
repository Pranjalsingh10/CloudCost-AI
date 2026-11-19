
from pathlib import Path

import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from mlflow import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT_DIR = Path(__file__).resolve().parent

MLFLOW_DB = ROOT_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

mlflow.set_experiment("CloudCost-AI-CPU-Forecasting")

MODEL_REGISTRY_NAME = "CloudCostAI_CPU_Forecaster"


def get_models():
    return {
        "LinearRegression": LinearRegression(),

        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
    }


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def train_models(X_train, X_test, y_train, y_test):
    results = []
    model_uris = {}

    for model_name, model in get_models().items():

        with mlflow.start_run(run_name=model_name) as run:

            model.fit(X_train, y_train)

            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            mlflow.log_params({
                "model_name": model_name,
                "number_of_features": X_train.shape[1],
                "training_rows": X_train.shape[0],
                "testing_rows": X_test.shape[0],
            })

            mlflow.log_metrics(metrics)

            signature = infer_signature(
                X_train,
                model.predict(X_train.head(5))
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5)
            )

            model_uri = f"runs:/{run.info.run_id}/model"

            model_uris[model_name] = model_uri

            results.append({
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Run_ID": run.info.run_id,
            })

            print(f"\n{model_name}")
            print("-" * 30)

            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    return results, model_uris




def register_best_model(results, model_uris):
    best_model = min(
        results,
        key=lambda result: result["RMSE"]
    )

    best_name = best_model["Model"]
    best_uri = model_uris[best_name]

    print("\nBest model:", best_name)
    print("Best RMSE:", best_model["RMSE"])

    registered_model = mlflow.register_model(
        model_uri=best_uri,
        name=MODEL_REGISTRY_NAME
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME,
        alias="champion",
        version=registered_model.version
    )

    print("\nModel registered successfully!")
    print("Registered model:", registered_model.name)
    print("Model version:", registered_model.version)
    print("Champion alias assigned to version:", registered_model.version)

    return registered_model




def print_results(results):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    sorted_results = sorted(
        results,
        key=lambda result: result["RMSE"]
    )

    for result in sorted_results:
        print(
            f"\n{result['Model']}\n"
            f"MAE:  {result['MAE']:.4f}\n"
            f"RMSE: {result['RMSE']:.4f}\n"
            f"R2:   {result['R2']:.4f}"
        )


def run_training(X_train, X_test, y_train, y_test):
    results, model_uris = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print_results(results)

    registered_model = register_best_model(
        results,
        model_uris
    )

    return results, registered_model
