
def analyze_cpu(predicted_cpu: float) -> dict:
    """
    Analyze predicted CPU utilization and return
    a resource optimization recommendation.
    """

    predicted_cpu = float(predicted_cpu)

    if not 0 <= predicted_cpu <= 100:
        raise ValueError("CPU utilization must be between 0 and 100")

    if predicted_cpu < 20:
        status = "UNDERUTILIZED"
        recommendation = (
            "Investigate whether the VM is over-provisioned "
            "and could use fewer allocated resources."
        )

    elif predicted_cpu < 70:
        status = "HEALTHY"
        recommendation = (
            "Resource utilization is within a healthy range. "
            "No immediate action is required."
        )

    elif predicted_cpu <= 85:
        status = "HIGH"
        recommendation = (
            "Monitor workload growth and prepare for possible "
            "capacity adjustments."
        )

    else:
        status = "CRITICAL"
        recommendation = (
            "High predicted CPU utilization. Consider scaling "
            "resources or redistributing the workload."
        )

    return {
        "predicted_cpu": round(predicted_cpu, 2),
        "status": status,
        "recommendation": recommendation
    }
