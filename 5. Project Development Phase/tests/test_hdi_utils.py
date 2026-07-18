"""
tests/test_hdi_utils.py

Unit tests for the HDI calculation logic in hdi_utils.py.
Run with: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hdi_utils import (
    calculate_hdi,
    classify_hdi,
    life_expectancy_index,
    education_index,
    income_index,
)


def test_life_expectancy_index_bounds():
    assert life_expectancy_index(20) == 0.0
    assert life_expectancy_index(85) == 1.0
    assert 0.0 <= life_expectancy_index(150) <= 1.0  # clamped, not > 1
    assert life_expectancy_index(0) == 0.0  # clamped, not negative


def test_education_index_bounds():
    assert education_index(0, 0) == 0.0
    assert education_index(15, 18) == 1.0
    assert 0.0 <= education_index(20, 25) <= 1.0  # clamped


def test_income_index_bounds():
    assert income_index(100) == 0.0
    assert income_index(75000) == 1.0
    assert income_index(50) == 0.0  # clamped at floor


def test_classify_hdi_bands():
    assert classify_hdi(0.95) == "Very High"
    assert classify_hdi(0.80) == "Very High"
    assert classify_hdi(0.75) == "High"
    assert classify_hdi(0.70) == "High"
    assert classify_hdi(0.60) == "Medium"
    assert classify_hdi(0.55) == "Medium"
    assert classify_hdi(0.40) == "Low"
    assert classify_hdi(0.0) == "Low"


def test_scenario_very_high_development():
    result = calculate_hdi(82.0, 12.5, 16.5, 55000)
    assert result["hdi_category"] == "Very High"
    assert result["hdi_score"] >= 0.800


def test_scenario_medium_development():
    result = calculate_hdi(68.0, 7.5, 11.0, 8000)
    assert result["hdi_category"] == "Medium"
    assert 0.550 <= result["hdi_score"] < 0.700


def test_scenario_low_development():
    result = calculate_hdi(55.0, 3.5, 6.0, 1200)
    assert result["hdi_category"] == "Low"
    assert result["hdi_score"] < 0.550


def test_result_keys_present():
    result = calculate_hdi(70, 8, 12, 15000)
    expected_keys = {
        "life_expectancy_index", "education_index",
        "income_index", "hdi_score", "hdi_category",
    }
    assert expected_keys == set(result.keys())
