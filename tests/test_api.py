import pytest

import api


@pytest.fixture
def client(monkeypatch):
    # Never call the real LLM from tests
    monkeypatch.setattr(api, "generate_recommendation",
                        lambda equipment, prediction: "Inspect the bearing. Priority: HIGH")
    return api.create_app().test_client()


def test_index_lists_the_endpoints_and_features(client):
    body = client.get("/").get_json()

    assert "POST /predict" in body["endpoints"]
    assert body["features"] == api.FEATURES


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy"}


def test_predict_returns_prediction_and_recommendation(client, high_risk_equipment):
    response = client.post("/predict", json=high_risk_equipment)
    body = response.get_json()

    assert response.status_code == 200
    assert body["prediction"]["risk_level"] == "HIGH"
    assert body["recommendation"] == "Inspect the bearing. Priority: HIGH"


def test_recommend_false_skips_the_llm(monkeypatch, high_risk_equipment):
    def fail(*_):
        raise AssertionError("the LLM must not be called")

    monkeypatch.setattr(api, "generate_recommendation", fail)
    client = api.create_app().test_client()

    response = client.post("/predict?recommend=false", json=high_risk_equipment)

    assert response.status_code == 200
    assert response.get_json()["recommendation"] is None


def test_llm_failure_still_returns_the_prediction(monkeypatch, high_risk_equipment):
    def fail(*_):
        raise RuntimeError("no API key")

    monkeypatch.setattr(api, "generate_recommendation", fail)
    client = api.create_app().test_client()

    response = client.post("/predict", json=high_risk_equipment)
    body = response.get_json()

    assert response.status_code == 200
    assert body["prediction"]["risk_level"] == "HIGH"
    assert body["recommendation"] is None
    assert "recommendation_error" in body


def test_extra_fields_are_ignored(client, high_risk_equipment):
    response = client.post("/predict", json={**high_risk_equipment, "equipment_id": "PUMP-0001"})

    assert response.status_code == 200
    assert "equipment_id" not in response.get_json()["equipment"]


def test_missing_body_is_rejected(client):
    response = client.post("/predict", data="not json", content_type="text/plain")

    assert response.status_code == 400


def test_non_object_body_is_rejected(client):
    response = client.post("/predict", json=[1, 2, 3])

    assert response.status_code == 422


def test_missing_fields_are_listed(client, high_risk_equipment):
    del high_risk_equipment["vibration_mm_s"]
    del high_risk_equipment["pressure_bar"]

    response = client.post("/predict", json=high_risk_equipment)
    details = response.get_json()["details"]

    assert response.status_code == 422
    assert any("vibration_mm_s" in d and "pressure_bar" in d for d in details)


@pytest.mark.parametrize("field, value, message", [
    ("temperature_c", "hot", "must be a number"),
    ("operating_hours", True, "must be a number"),
    ("operating_hours", None, "must be a number"),
    ("vibration_mm_s", -1.0, "must not be negative"),
    ("previous_failures", 1.5, "must be a whole number"),
])
def test_invalid_values_are_rejected(client, high_risk_equipment, field, value, message):
    high_risk_equipment[field] = value

    response = client.post("/predict", json=high_risk_equipment)

    assert response.status_code == 422
    assert any(field in d and message in d for d in response.get_json()["details"])


def test_negative_temperature_is_allowed(client, low_risk_equipment):
    low_risk_equipment["temperature_c"] = -5

    response = client.post("/predict", json=low_risk_equipment)

    assert response.status_code == 200
