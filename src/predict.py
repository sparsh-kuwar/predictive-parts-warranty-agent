import joblib
import pandas as pd


# Load trained model
model = joblib.load(
    "models/failure_prediction_model.pkl"
)


FEATURES = [
    "equipment_age_years",
    "operating_hours",
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "maintenance_count",
    "days_since_maintenance",
    "part_age_hours",
    "previous_failures"
]


def predict_failure(equipment):

    # Convert input into DataFrame
    data = pd.DataFrame(
        [equipment],
        columns=FEATURES
    )

    # Prediction
    prediction = model.predict(data)[0]

    # Failure probability
    probability = model.predict_proba(data)[0][1]

    # Risk classification
    if probability >= 0.70:
        risk = "HIGH"
    elif probability >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "failure_predicted": bool(prediction),
        "failure_probability": round(
            float(probability),
            3
        ),
        "risk_level": risk
    }


if __name__ == "__main__":

    # Example equipment
    equipment = {
        "equipment_age_years": 9.5,
        "operating_hours": 10500,
        "temperature_c": 91,
        "vibration_mm_s": 6.5,
        "pressure_bar": 3.1,
        "maintenance_count": 2,
        "days_since_maintenance": 280,
        "part_age_hours": 4200,
        "previous_failures": 3
    }

    result = predict_failure(equipment)

    print("\n==============================")
    print("FAILURE PREDICTION")
    print("==============================")

    print(
        f"\nFailure Predicted: "
        f"{result['failure_predicted']}"
    )

    print(
        f"Failure Probability: "
        f"{result['failure_probability'] * 100:.1f}%"
    )

    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )