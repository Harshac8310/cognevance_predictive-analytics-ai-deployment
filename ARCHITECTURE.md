# System Architecture

## Components
1. **Data layer** — CSV dataset under `data/raw/`; the default dataset is synthetic.
2. **Training layer** — `src/train.py` creates a preprocessing + estimator pipeline and compares three classifiers.
3. **Artifact layer** — best fitted pipeline and metadata are stored in `models/`.
4. **Evaluation layer** — CSV metrics, classification report, ROC curves, confusion matrix, and feature importance are stored in `reports/`.
5. **Inference API** — FastAPI validates customer input, loads the trained pipeline, and returns prediction/probability.
6. **Dashboard** — Streamlit reads the dataset and reports, and uses the saved pipeline for interactive predictions.

## Data flow
```mermaid
flowchart TD
    A[CSV Dataset] --> B[Data Validation]
    B --> C[Train/Test Split]
    C --> D[Imputation + Encoding + Scaling]
    D --> E1[Logistic Regression]
    D --> E2[Random Forest]
    D --> E3[Gradient Boosting]
    E1 --> F[Evaluate on Held-out Test Set]
    E2 --> F
    E3 --> F
    F --> G[Select Model by ROC-AUC]
    G --> H[Save Pipeline with Joblib]
    H --> I[FastAPI /predict]
    H --> J[Streamlit Dashboard]
    F --> K[Reports and Charts]
    K --> J
```

## Reproducibility and leakage prevention
The preprocessing transformer is part of each scikit-learn pipeline. It is fitted on training data only, and then applied to the test data. The test set is not used to fit the estimators.
