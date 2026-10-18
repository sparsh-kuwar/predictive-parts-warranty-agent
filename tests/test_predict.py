from predict import FEATURES, predict_failure


def test_high_risk_equipment_is_flagged(high_risk_equipment):
    result = predict_failure(high_risk_equipment)

    assert result["failure_predicted"] is True
    assert result["risk_level"] == "HIGH"


def test_low_risk_equipment_is_not_flagged(low_risk_equipment):
    result = predict_failure(low_risk_equipment)

    assert result["failure_predicted"] is False
    assert result["risk_level"] == "LOW"


def test_probability_is_between_zero_and_one(high_risk_equipment, low_risk_equipment):
    for equipment in (high_risk_equipment, low_risk_equipment):
        probability = predict_failure(equipment)["failure_probability"]
        assert 0.0 <= probability <= 1.0


def test_risk_level_matches_thresholds(high_risk_equipment, low_risk_equipment):
    for equipment in (high_risk_equipment, low_risk_equipment):
        result = predict_failure(equipment)
        p = result["failure_probability"]
        expected = "HIGH" if p >= 0.70 else "MEDIUM" if p >= 0.40 else "LOW"
        assert result["risk_level"] == expected


def test_input_key_order_does_not_change_the_result(high_risk_equipment):
    reversed_input = dict(reversed(list(high_risk_equipment.items())))

    assert predict_failure(reversed_input) == predict_failure(high_risk_equipment)


def test_feature_list_matches_the_trained_model():
    from predict import model

    assert list(model.feature_names_in_) == FEATURES
