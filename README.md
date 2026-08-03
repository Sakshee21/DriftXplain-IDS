# DriftXplain-IDS

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://driftxplain-ids-hnewqv9ezpn3nbwowkzqiu.streamlit.app/)

**Live demo:** https://driftxplain-ids-hnewqv9ezpn3nbwowkzqiu.streamlit.app/

An Intrusion Detection System (IDS) for CICIDS2017 with:

- End-to-end preprocessing and drift-aware splitting
- Feature selection and model comparison
- Final Random Forest inference pipeline
- SHAP local and summary explainability in Streamlit

## What Is In This Repository

This project has two major notebook phases and one deployable app:

1. Preprocessing notebook:
	- Data loading from CICIDS2017 CSV files
	- Label cleanup (including Web Attack naming fixes)
	- Numeric cleaning (inf/NaN handling)
	- Outlier clipping (IQR)
	- Main vs drift split (Mon-Wed for train/val/test, Thu-Fri for drift)
	- MinMax scaling and split export

2. Model comparison and final training notebook:
	- Correlation-based top-20 feature selection
	- Multiple model baselines (LR, RF, XGB)
	- Imbalance handling experiments (SMOTE, class-weighted)
	- Final model artifact saving (rf_final.pkl, scaler.pkl, features.pkl)
	- SHAP analysis (summary, bar, waterfall, dependence)

3. Streamlit app:
	- Interactive inference with threshold tuning
	- One-click demo presets from tests/demo_inputs.csv
	- SHAP waterfall + top positive/negative contributions
	- SHAP summary plot for global pattern view

## Project Structure

```text
DriftXplain-IDS/
├─ app/
│  └─ app.py
├─ models/
│  ├─ rf_final.pkl
│  ├─ scaler.pkl
│  └─ features.pkl
├─ notebooks/
│  ├─ IDS-Preprocessing.ipynb
│  └─ IDS-Model comparsions and Final Model Training with SHAP.ipynb
├─ tests/
│  ├─ demo_inputs.csv
│  ├─ generate_demo_inputs.py
│  └─ run_demo_cases.py
├─ requirements.txt
└─ README.md
```

## Local Setup

Use Python 3.12 so scikit-learn 1.6.1 wheels install cleanly.

1. Create and activate environment:

```bash
py -3.12 -m venv venv312
venv312\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify critical version:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

Expected output: 1.6.1

## Run Streamlit App

Hosted version (no setup required): https://driftxplain-ids-hnewqv9ezpn3nbwowkzqiu.streamlit.app/

Or run locally:

```bash
streamlit run app/app.py
```

Inside the app:

- Use the sidebar threshold slider (default 0.40)
- Load one-click demo presets
- Click Predict to view probability and SHAP explanations

## Terminal Testing (No Manual UI Entry)

Run scripted demo predictions:

```bash
venv312\Scripts\python.exe tests/run_demo_cases.py
```

Regenerate demo rows from current model:

```bash
venv312\Scripts\python.exe tests/generate_demo_inputs.py
```

## Notebook Usage Notes

The notebooks currently include Google Drive paths (for Colab), for example:

- /content/drive/MyDrive/ids_project/data/processed/
- /content/drive/MyDrive/ids_project/models/

If running locally, update these paths to project-relative locations.

Recommended local mapping:

- processed data -> data/processed/
- models -> models/
- report outputs -> reports/

## Current Inference Contract

- Model type: RandomForestClassifier
- Feature count: 20 (must match scaler and feature artifact)
- Default attack threshold: 0.40
- Explainability: SHAP waterfall + contribution tables + SHAP summary plot