import logging
import math

from flask import Flask, jsonify, request

from predict import FEATURES, predict_failure
from warranty_agent import generate_recommendation


log = logging.getLogger(__name__)

# Features that count events, so they must be whole numbers
INTEGER_FEATURES = {
    "maintenance_count",
    "days_since_maintenance",
    "previous_failures"
}

# temperature_c may be negative; every other feature is a count, an age or a
# measurement that cannot go below zero
NON_NEGATIVE_FEATURES = set(FEATURES) - {"temperature_c"}


def validate_equipment(data):
    """Return (equipment, errors). equipment holds only the model features."""

    if not isinstance(data, dict):
        return None, ["Request body must be a JSON object"]

    errors = []
    equipment = {}

    missing = [f for f in FEATURES if f not in data]
    if missing:
        errors.append("Missing fields: " + ", ".join(missing))

    for feature in FEATURES:
        if feature not in data:
            continue
        value = data[feature]

        # bool is a subclass of int in Python, so reject it explicitly
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{feature} must be a number")
            continue
        if not math.isfinite(value):
            errors.append(f"{feature} must be a finite number")
            continue
        if feature in INTEGER_FEATURES and value != int(value):
            errors.append(f"{feature} must be a whole number")
            continue
        if feature in NON_NEGATIVE_FEATURES and value < 0:
            errors.append(f"{feature} must not be negative")
            continue

        equipment[feature] = value

    return equipment, errors


def create_app():

    app = Flask(__name__)
    app.json.sort_keys = False

    @app.get("/")
    def index():
        # Opening the deployed URL in a browser shows how to use the API
        return {
            "service": "Predictive Parts & Warranty Agent",
            "endpoints": {
                "GET /health": "Liveness check",
                "POST /predict": "JSON body with the 9 equipment features; "
                                 "add ?recommend=false to skip the LLM"
            },
            "features": FEATURES
        }

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Send a JSON body with the equipment measurements"}), 400

        equipment, errors = validate_equipment(data)
        if errors:
            return jsonify({"error": "Invalid input", "details": errors}), 422

        prediction = predict_failure(equipment)
        response = {
            "equipment": equipment,
            "prediction": prediction,
            "recommendation": None
        }

        # ?recommend=false returns only the ML prediction and skips the LLM
        if request.args.get("recommend", "true").lower() == "false":
            return response

        # The prediction is still useful when the LLM is unavailable
        # (no API key, network error, rate limit), so never fail the request
        try:
            response["recommendation"] = generate_recommendation(equipment, prediction)
        except Exception as exc:
            log.warning("LLM recommendation failed: %s", exc)
            response["recommendation_error"] = (
                "The recommendation service is unavailable; the ML prediction is still valid."
            )

        return response

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
