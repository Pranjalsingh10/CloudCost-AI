
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src.inference import predict_cpu
from src.optimization import analyze_cpu

app = FastAPI(
    title="CloudCost AI",
    description="CPU forecasting and cloud resource optimization API",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    vm_id: str = Field(default="vm-101", min_length=1)
    cpu_history: list[float] = Field(min_length=12)

    @field_validator("cpu_history")
    @classmethod
    def validate_cpu_history(cls, values):
        if any(not 0 <= value <= 100 for value in values):
            raise ValueError("CPU values must be between 0 and 100")

        return values


@app.get("/")
def root():
    return {
        "application": "CloudCost AI",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "CloudCostAI_CPU_Forecaster",
        "alias": "champion"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    predicted_cpu = predict_cpu(request.cpu_history)

    return {
        "vm_id": request.vm_id,
        "predicted_cpu": round(predicted_cpu, 2),
        "unit": "percentage"
    }


@app.post("/optimize")
def optimize(request: PredictionRequest):
    predicted_cpu = predict_cpu(request.cpu_history)

    result = analyze_cpu(predicted_cpu)

    return {
        "vm_id": request.vm_id,
        **result
    }
