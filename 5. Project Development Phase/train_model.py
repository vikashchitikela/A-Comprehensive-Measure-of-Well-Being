"""
train_model.py

Generates a synthetic, HDI-formula-consistent dataset and trains two models:
  1. A RandomForestRegressor  -> predicts the continuous HDI score
  2. A RandomForestClassifier -> predicts the HDI tier (Very High/High/Medium/Low)

Both models are saved to the models/ directory via joblib so the Streamlit
app can load them without retraining on every run.

Run:
    python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import joblib
import os

from hdi_utils import calculate_hdi

RANDOM_STATE = 42
N_SAMPLES = 5000
OUTPUT_DIR = "models"


def generate_synthetic_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Build a synthetic dataset of plausible country-level indicators, spanning
    the full realistic range of each variable, and label each row using the
    true HDI formula (hdi_utils.calculate_hdi) so the model learns a function
    that is consistent with the official methodology rather than noise.
    """
    rng = np.random.default_rng(seed)

    life_expectancy = rng.uniform(45, 85, n_samples)
    mean_years_schooling = rng.uniform(1, 15, n_samples)
    expected_years_schooling = rng.uniform(4, 18, n_samples)
    gni_per_capita = rng.lognormal(mean=8.5, sigma=1.1, size=n_samples)
    gni_per_capita = np.clip(gni_per_capita, 300, 120000)

    rows = []
    for le, mys, eys, gni in zip(life_expectancy, mean_years_schooling,
                                  expected_years_schooling, gni_per_capita):
        result = calculate_hdi(le, mys, eys, gni)
        rows.append({
            "life_expectancy": le,
            "mean_years_schooling": mys,
            "expected_years_schooling": eys,
            "gni_per_capita": gni,
            "hdi_score": result["hdi_score"],
            "hdi_category": result["hdi_category"],
        })

    return pd.DataFrame(rows)


def train_and_save_models(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = ["life_expectancy", "mean_years_schooling",
                     "expected_years_schooling", "gni_per_capita"]
    X = df[feature_cols]
    y_reg = df["hdi_score"]
    y_clf = df["hdi_category"]

    X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE
    )

    # --- Regressor: predicts the continuous HDI score ---
    regressor = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_STATE)
    regressor.fit(X_train, yreg_train)
    reg_preds = regressor.predict(X_test)
    print(f"[Regressor]  R^2 on holdout set: {r2_score(yreg_test, reg_preds):.4f}")

    # --- Classifier: predicts the HDI tier label ---
    classifier = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE)
    classifier.fit(X_train, yclf_train)
    clf_preds = classifier.predict(X_test)
    print(f"[Classifier] Accuracy on holdout set: {accuracy_score(yclf_test, clf_preds):.4f}")

    joblib.dump(regressor, os.path.join(output_dir, "hdi_regressor.joblib"))
    joblib.dump(classifier, os.path.join(output_dir, "hdi_classifier.joblib"))
    joblib.dump(feature_cols, os.path.join(output_dir, "feature_columns.joblib"))

    print(f"Models saved to '{output_dir}/'")


if __name__ == "__main__":
    print("Generating synthetic training dataset...")
    dataset = generate_synthetic_dataset()
    dataset.to_csv("data_sample.csv", index=False)
    print(f"Dataset shape: {dataset.shape}")
    print(dataset["hdi_category"].value_counts())

    print("\nTraining models...")
    train_and_save_models(dataset)
