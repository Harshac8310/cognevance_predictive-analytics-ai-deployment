# Customer Churn Prediction AI
### Cognevance Level 3 — Predictive Analytics & AI Model Deployment

An end-to-end machine-learning demonstration that trains and compares customer churn classifiers, saves a reusable preprocessing/model pipeline, exposes predictions through FastAPI, and provides a Streamlit analytics dashboard.

> **Data transparency:** The project generates a synthetic telecom-style dataset by default so that the full pipeline can run immediately without downloading private or licensed data. Synthetic results are for demonstrating the workflow only and must not be represented as real-world model performance. For a real-data extension, use the public IBM Telco Customer Churn dataset and follow its current license/terms.

## Features
- Reproducible demo dataset generation
- Missing-value imputation, categorical encoding, numeric scaling
- Stratified train/test split
- Logistic Regression, Random Forest, and Gradient Boosting comparison
- Accuracy, precision, recall, F1-score, ROC-AUC, classification report
- Confusion matrix, ROC curves, model comparison and feature-importance charts
- Persisted scikit-learn pipeline (`models/best_model.joblib`)
- FastAPI `/predict`, `/health`, and `/docs` endpoints
- Streamlit dashboard with exploratory charts and interactive predictions
- Basic API tests

## Project architecture
```text
CSV data -> preprocessing pipeline -> train/compare models -> evaluation reports
                                      |
                                      v
                                saved model
                                  /     \
                           FastAPI API  Streamlit dashboard
```

## Setup (Python 3.10–3.12 recommended)
Python 3.13 may work depending on package wheels; Python 3.11 is a safe choice.

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.generate_demo_data
python -m src.train
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.generate_demo_data
python -m src.train
```

Training creates:
- `models/best_model.joblib`
- `models/model_metadata.json`
- `reports/model_comparison.csv`
- `reports/classification_report.txt`
- `reports/figures/*.png`
- `reports/feature_importance.csv` (when supported)

## Run the API
From the repository root, in one terminal:
```bash
uvicorn api.main:app --reload
```
Open:
- API landing page: http://127.0.0.1:8000/
- Interactive API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Example JSON body for `POST /predict`:
```json
{
  "tenure": 3,
  "monthly_charges": 120.0,
  "total_charges": 360.0,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "tech_support": "No",
  "payment_method": "Electronic check",
  "paperless_billing": "Yes",
  "senior_citizen": 0
}
```

## Run the dashboard
In another terminal, with the virtual environment activated:
```bash
streamlit run dashboard/app.py
```
Streamlit displays dataset summaries, churn distribution, churn by contract, model metrics, and a form for individual predictions.

## Tests
```bash
pytest -q
```
The health and root tests do not require a trained model. For `/predict`, train first.

## Dataset schema
The generated dataset contains customer_id, tenure, monthly_charges, total_charges, contract, internet_service, tech_support, payment_method, paperless_billing, senior_citizen, and churn. `customer_id` is dropped before training. `churn` is the target.

## Evaluation methodology
- A stratified 80/20 train/test split is used.
- Preprocessing is fitted inside each model pipeline using training data only.
- The models are compared on the same held-out test set.
- ROC-AUC is used as the default model-selection criterion; also inspect recall and precision for the business use case.
- Metrics vary by generated dataset seed and must be taken from the actual run's reports.

## Important limitations
- The included data is synthetic and not representative of any real company.
- This is a learning/demo application, not a validated production system.
- A real deployment requires external validation, monitoring, security hardening, drift checks, calibrated probabilities, privacy review, and business approval.
- Churn risk is not certainty; do not use predictions as the sole basis for decisions affecting people.

## Optional real dataset
To extend the project, obtain the IBM Telco Customer Churn CSV from a legitimate public source, review its terms, and adapt the schema mapping in `src/train.py` and `dashboard/app.py`. Do not claim the synthetic demo metrics are results from the IBM dataset.

## GitHub submission checklist
1. Create repository `cognevance_predictive-analytics-ai-deployment`.
2. Include source code, README, report, and screenshots from your own run.
3. Run training and both applications locally; capture screenshots.
4. Add actual evaluation metrics to your report.
5. Review all files and push to GitHub.
6. Email the repository URL and your honest experience/review to `support@cognevance.online`.
