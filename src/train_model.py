import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# Load dataset
df = pd.read_csv("data/equipment_failure_data.csv")


# Features used by the model
features = [
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

X = df[features]
y = df["failure_next_30_days"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)


# Train
print("Training Random Forest model...")

model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\nAccuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# Save model
joblib.dump(
    model,
    "models/failure_prediction_model.pkl"
)

print("\nModel saved successfully!")
print("models/failure_prediction_model.pkl")