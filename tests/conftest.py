import sys
from pathlib import Path

import pytest

# The modules in src/ import each other by name (from predict import ...),
# the same way they do when run with python src/main.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def high_risk_equipment():
    return {
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


@pytest.fixture
def low_risk_equipment():
    return {
        "equipment_age_years": 1.0,
        "operating_hours": 800,
        "temperature_c": 58,
        "vibration_mm_s": 1.2,
        "pressure_bar": 3.0,
        "maintenance_count": 4,
        "days_since_maintenance": 20,
        "part_age_hours": 300,
        "previous_failures": 0
    }
