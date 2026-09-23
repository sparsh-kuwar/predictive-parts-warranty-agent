from predict import predict_failure
from warranty_agent import generate_recommendation


def main():

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

    print("\n================================")
    print("PREDICTIVE PARTS & WARRANTY AGENT")
    print("================================")

    print("\nRunning ML failure prediction...")

    prediction = predict_failure(equipment)

    print(
        f"\nFailure Probability: "
        f"{prediction['failure_probability'] * 100:.1f}%"
    )

    print(
        f"Risk Level: "
        f"{prediction['risk_level']}"
    )

    print("\nGenerating maintenance recommendation...")

    recommendation = generate_recommendation(
        equipment,
        prediction
    )

    print("\n================================")
    print("AI RECOMMENDATION")
    print("================================\n")

    print(recommendation)


if __name__ == "__main__":
    main()