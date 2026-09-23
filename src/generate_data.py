import pandas as pd
import numpy as np


np.random.seed(42)

NUM_SAMPLES = 1000


data = {
    "equipment_id": [
        f"PUMP-{i:04d}"
        for i in range(1, NUM_SAMPLES + 1)
    ],

    "equipment_age_years": np.random.uniform(
        0.5, 12, NUM_SAMPLES
    ),

    "operating_hours": np.random.uniform(
        500, 12000, NUM_SAMPLES
    ),

    "temperature_c": np.random.uniform(
        55, 100, NUM_SAMPLES
    ),

    "vibration_mm_s": np.random.uniform(
        1.0, 8.0, NUM_SAMPLES
    ),

    "pressure_bar": np.random.uniform(
        2.0, 5.5, NUM_SAMPLES
    ),

    "maintenance_count": np.random.randint(
        0, 15, NUM_SAMPLES
    ),

    "days_since_maintenance": np.random.randint(
        1, 500, NUM_SAMPLES
    ),

    "part_age_hours": np.random.uniform(
        100, 5000, NUM_SAMPLES
    ),

    "previous_failures": np.random.randint(
        0, 5, NUM_SAMPLES
    )
}


df = pd.DataFrame(data)


# ---------------------------------------
# Generate realistic failure probability
# ---------------------------------------

risk_score = (
    0.04 * df["equipment_age_years"]
    + 0.00012 * df["operating_hours"]
    + 0.035 * (df["temperature_c"] - 70)
    + 0.25 * df["vibration_mm_s"]
    + 0.00025 * df["part_age_hours"]
    + 0.12 * df["previous_failures"]
    + 0.0015 * df["days_since_maintenance"]
)


# Add randomness
risk_score += np.random.normal(
    0,
    0.8,
    NUM_SAMPLES
)


# Convert risk score to probability
failure_probability = 1 / (
    1 + np.exp(-risk_score + 5)
)


# Generate binary target
df["failure_next_30_days"] = (
    failure_probability > 0.5
).astype(int)


# Save dataset
df.to_csv(
    "data/equipment_failure_data.csv",
    index=False
)


print("Dataset created successfully!")
print(f"Rows: {len(df)}")
print("\nFailure distribution:")
print(
    df["failure_next_30_days"]
    .value_counts()
)

print("\nSample records:")
print(df.head())