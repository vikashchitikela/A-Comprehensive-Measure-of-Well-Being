# HDI Prediction System — Human Development Intelligence

Predict and analyze the Human Development Index (HDI) using machine learning.
Evaluate a country's development level based on life expectancy, education,
and income indicators.

The Human Development Index (HDI) is a statistical composite index of life
expectancy, education, and per-capita income indicators, used to rank
countries into four tiers — **Very High, High, Medium, and Low** — of human
development.

## Features

- Formula-based HDI calculation following the official UNDP methodology
  (geometric mean of the Life Expectancy, Education, and Income indices)
- ML-based prediction using a trained RandomForest classifier/regressor pair
- Interactive Streamlit interface with sliders and preset scenarios
- Three built-in demo scenarios matching common development profiles:
  - **Very High** — developed nation
  - **Medium** — emerging economy
  - **Low** — needs development intervention

## Project Structure

```
hdi-prediction-system/
├── app.py                    # Streamlit application entry point
├── hdi_utils.py               # UNDP HDI formula implementation
├── train_model.py             # Generates training data & trains ML models
├── requirements.txt            # Python dependencies
├── .env.example                # Environment configuration template
├── .gitignore
├── models/                     # Trained model artifacts (generated)
│   ├── hdi_regressor.joblib
│   ├── hdi_classifier.joblib
│   └── feature_columns.joblib
└── tests/
    └── test_hdi_utils.py       # Unit tests for the HDI formula
```

## Pre-requisites

- Python 3.10+
- pip

## Setup & Run Instructions

1. **Clone the repository / extract the project files**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment (optional)**
   ```bash
   cp .env.example .env
   ```

4. **Train the models** (generates `models/*.joblib` — only needed once,
   or whenever you want to retrain)
   ```bash
   python train_model.py
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** at the URL Streamlit prints (typically
   `http://localhost:8501`)

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Methodology

```
HDI = (LEI × EI × II) ^ (1/3)

LEI (Life Expectancy Index) = (LE − 20) / (85 − 20)
EI  (Education Index)       = (MYS/15 + EYS/18) / 2
II  (Income Index)          = (ln(GNIpc) − ln(100)) / (ln(75000) − ln(100))
```

**Classification bands:**

| Tier       | HDI Score     |
|------------|---------------|
| Very High  | ≥ 0.800       |
| High       | 0.700 – 0.799 |
| Medium     | 0.550 – 0.699 |
| Low        | < 0.550       |

## Known Issues / Limitations

- The ML models are trained on synthetically generated data consistent with
  the UNDP formula, not on real historical country-level HDI datasets — for
  production use, retrain on official UNDP Human Development Report data.
- No persistent database; each session is stateless.

## License

For educational and research purposes.
