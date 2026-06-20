# Nigeria Agricultural Produce Price Predictor
### Final Year Project — Software Engineering

---

## Overview

A machine learning web application that predicts the market price (in NGN per kg)
of 7 major agricultural commodities across Nigerian states and markets.

**Live inputs:** Crop type, State, Market, Year, Month  
**Output:** Predicted price per kg + per 50kg + per 100kg bag  
**Model:** Random Forest (best performer, 76–91% accuracy across commodities)

---

## Commodities Covered

| Crop | Model Accuracy |
|---|---|
| Maize (White) | 90.69% |
| Maize (Yellow) | 89.70% |
| Yam | 85.52% |
| Cassava Meal / Gari (Yellow) | 84.61% |
| Gari (White) | 82.81% |
| Rice (Imported) | 77.79% |
| Rice (Local) | 76.01% |

---

## How to Run Locally

### 1. Clone / download the project files

Ensure the following files are in the same folder:

```
app.py
dataset_engineered.csv
requirements.txt
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## How to Deploy on Streamlit Community Cloud (Free)

1. Push your project folder to a **GitHub repository** (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy** — the app will be live at a public URL within 2–3 minutes

> Make sure `dataset_engineered.csv` is committed to the repository.  
> Streamlit Cloud reads `requirements.txt` automatically.

---

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `dataset_engineered.csv` | Engineered training dataset (3,412 rows × 93 cols) |
| `requirements.txt` | Python dependencies |
| `merged_final_dataset.csv` | Pre-imputation merged dataset |
| `merged_final_dataset_imputed.csv` | Post-imputation dataset |
| `model_results.json` | Evaluation metrics for all 5 models × 7 commodities |
| `model_predictions.csv` | Actual vs predicted values for the test period |
| `label_encodings.json` | State and market label encoding mappings |
| `merging_pipeline_documentation.md` | Data merging methodology |
| `imputation_documentation.md` | Missing value handling methodology |
| `feature_engineering_documentation.md` | Feature engineering + model training |
| `model_training_documentation.md` | Full model results and interpretation |

---

## Data Sources

- **WFP Nigeria Food Prices** — World Food Programme VAM portal
  (monthly commodity prices, 2002–2026, 58 markets)
- **Macro dataset** — CPI, fuel price, temperature, rainfall per market per year

---

## Model Architecture

```
Input: Crop, State, Market, Year, Month
    ↓
Feature construction:
  - Temporal: year, month (sin/cos encoded)
  - Macro: CPI, fuel price, temperature, rainfall, CPI×fuel
  - Geographic: latitude, longitude, state encoding, market encoding
  - Price lags: lag-1, lag-3, lag-6, lag-12 months (from historical data)
  - Rolling stats: 3-month mean, 6-month mean, 3-month volatility
  - Inflation-adjusted lags: price/CPI × base_CPI
    ↓
Random Forest (300 trees, max_depth=12)
  - Trained on full 2009–2024 dataset per commodity
  - 22 features total
    ↓
Output: Predicted price per kg (NGN)
```

---

## Geographic Coverage

**14 States:** Abia, Adamawa, Borno, Gombe, Jigawa, Kaduna, Kano, Katsina, Kebbi,
Lagos, Oyo, Sokoto, Yobe, Zamfara

**58 Markets** — from major commodity hubs (Dawanau/Kano, Lagos, Ibadan/Oyo)
to humanitarian CBM markets (Maiduguri, Damaturu, Potiskum)

---

*Built as part of Final Year Project — Predictive Model of Agricultural Produce Pricing in Nigeria*
