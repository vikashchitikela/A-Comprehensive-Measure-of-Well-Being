"""
hdi_utils.py

Core domain logic for the Human Development Index (HDI) Prediction System.

Implements the UNDP HDI methodology:
    HDI = (LEI * EI * II) ** (1/3)

Where:
    LEI (Life Expectancy Index)  = (LE - 20)   / (85 - 20)
    EI  (Education Index)        = (MYS_index + EYS_index) / 2
        MYS_index = MYS / 15
        EYS_index = EYS / 18
    II  (Income Index)           = (ln(GNIpc) - ln(100)) / (ln(75000) - ln(100))

Classification tiers (UNDP standard bands):
    Very High : HDI >= 0.800
    High      : 0.700 <= HDI < 0.800
    Medium    : 0.550 <= HDI < 0.700
    Low       : HDI < 0.550
"""

import math

# ---- Normalization bounds (UNDP standard goalposts) ----
LE_MIN, LE_MAX = 20, 85
MYS_MAX = 15
EYS_MAX = 18
GNI_MIN, GNI_MAX = 100, 75000

HDI_BANDS = [
    ("Very High", 0.800),
    ("High", 0.700),
    ("Medium", 0.550),
    ("Low", 0.000),
]


def life_expectancy_index(life_expectancy: float) -> float:
    """Normalize life expectancy at birth (years) to [0, 1]."""
    value = (life_expectancy - LE_MIN) / (LE_MAX - LE_MIN)
    return min(max(value, 0.0), 1.0)


def education_index(mean_years_schooling: float, expected_years_schooling: float) -> float:
    """Normalize the two schooling indicators and average them to [0, 1]."""
    mys_index = min(max(mean_years_schooling / MYS_MAX, 0.0), 1.0)
    eys_index = min(max(expected_years_schooling / EYS_MAX, 0.0), 1.0)
    return (mys_index + eys_index) / 2


def income_index(gni_per_capita: float) -> float:
    """Normalize (log-scaled) GNI per capita (PPP $) to [0, 1]."""
    gni = max(gni_per_capita, GNI_MIN)
    value = (math.log(gni) - math.log(GNI_MIN)) / (math.log(GNI_MAX) - math.log(GNI_MIN))
    return min(max(value, 0.0), 1.0)


def calculate_hdi(life_expectancy: float, mean_years_schooling: float,
                   expected_years_schooling: float, gni_per_capita: float) -> dict:
    """
    Compute the HDI score and its three sub-indices for a given country's indicators.

    Returns a dict with the sub-indices, the composite HDI score (rounded to 3 dp),
    and the development tier label.
    """
    lei = life_expectancy_index(life_expectancy)
    ei = education_index(mean_years_schooling, expected_years_schooling)
    ii = income_index(gni_per_capita)

    # Geometric mean of the three sub-indices
    hdi_score = (lei * ei * ii) ** (1 / 3)
    hdi_score = round(hdi_score, 3)

    return {
        "life_expectancy_index": round(lei, 3),
        "education_index": round(ei, 3),
        "income_index": round(ii, 3),
        "hdi_score": hdi_score,
        "hdi_category": classify_hdi(hdi_score),
    }


def classify_hdi(hdi_score: float) -> str:
    """Map a composite HDI score to its UNDP development tier."""
    for label, threshold in HDI_BANDS:
        if hdi_score >= threshold:
            return label
    return "Low"


if __name__ == "__main__":
    # Quick sanity check against the three scenarios described in the project brief
    scenarios = {
        "Very High (developed nation)": (82.0, 12.5, 16.5, 55000),
        "Medium (emerging economy)": (68.0, 7.5, 11.0, 8000),
        "Low (needs intervention)": (55.0, 3.5, 6.0, 1200),
    }
    for name, (le, mys, eys, gni) in scenarios.items():
        result = calculate_hdi(le, mys, eys, gni)
        print(f"{name}: {result}")
