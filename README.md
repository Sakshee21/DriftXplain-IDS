# DriftXplain-IDS

## Setup

1. Create and activate Python 3.12 virtual environment.

```bash
py -3.12 -m venv venv312
venv312\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify sklearn version is 1.6.1:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

## Run App

```bash
streamlit run app/app.py
```

## Demo Test Cases

Run the scripted demo inputs without manual form entry:

```bash
venv312\Scripts\python.exe tests/run_demo_cases.py
```

Refresh demo rows automatically from the trained model:

```bash
venv312\Scripts\python.exe tests/generate_demo_inputs.py
```

Input rows are stored in tests/demo_inputs.csv.

## Notes

- The app loads model artifacts from the models directory relative to the project root.
- Threshold is configurable in the sidebar and defaults to 0.40.
- Prediction explanations are shown with SHAP waterfall plot and top feature contribution tables.