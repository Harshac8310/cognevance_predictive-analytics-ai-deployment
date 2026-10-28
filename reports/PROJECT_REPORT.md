# Project Report: Predictive Analytics & AI Model Deployment

## Abstract
This project demonstrates an end-to-end machine-learning workflow for customer churn risk estimation. A reproducible synthetic telecom dataset is generated, cleaned and transformed through a scikit-learn pipeline, and used to compare Logistic Regression, Random Forest and Gradient Boosting. The best model by held-out ROC-AUC is saved and exposed through FastAPI. A Streamlit dashboard provides dataset-level visualizations and interactive predictions.

## Problem statement
Customer churn can reduce recurring revenue. A predictive model can help prioritize which customers may need retention support. The model estimates likelihood; it does not determine customer intent with certainty.

## Objectives
- Generate or load structured customer data.
- Apply robust preprocessing and feature encoding.
- Compare multiple supervised learning algorithms.
- Evaluate performance using multiple metrics.
- Save the fitted model and make it available through an API.
- Provide a dashboard and document the workflow.

## Methodology
1. Generate a reproducible synthetic dataset.
2. Separate the target `churn` from input features and drop the identifier.
3. Split data into stratified 80% training and 20% test subsets.
4. Fit median imputation and standard scaling for numeric columns.
5. Fit most-frequent imputation and one-hot encoding for categorical columns.
6. Train Logistic Regression, Random Forest, and Gradient Boosting.
7. Calculate accuracy, precision, recall, F1-score and ROC-AUC on the held-out test set.
8. Select the model with the highest test ROC-AUC for demonstration and save the full pipeline.
9. Serve validated predictions through FastAPI and show analytics in Streamlit.

## Results
After running `python -m src.train`, use `model_comparison.csv`, `classification_report.txt`, and the figures in `reports/figures/` to report the actual results. Do not insert invented metrics. Since the default dataset is synthetic, these metrics demonstrate code execution only and are not evidence of real-world performance.

## Deployment
- API: FastAPI + Uvicorn (`uvicorn api.main:app --reload`)
- Dashboard: Streamlit (`streamlit run dashboard/app.py`)
- Saved model: Joblib artifact

## Limitations and ethics
Synthetic data does not capture real customer behavior. Before practical use, retrain and validate on appropriately licensed, representative data; check class balance and calibration; review privacy obligations; test performance across customer groups; and monitor drift. Predictions should support human judgment rather than automatically determine customer treatment.

## Conclusion
The project covers data preparation, model comparison, evaluation, model persistence, API inference, dashboard visualization, and technical documentation. It is an educational baseline that can be extended with real data and production monitoring.
