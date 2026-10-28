"""Train and compare churn classifiers; writes model, metrics, and figures."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, RocCurveDisplay)
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "customer_churn_demo.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TARGET = "churn"
DROP_COLS = ["customer_id"]

def load_data():
    if not DATA.exists():
        from src.generate_demo_data import main as generate
        generate()
    df = pd.read_csv(DATA)
    if TARGET not in df:
        raise ValueError(f"Dataset must include a '{TARGET}' target column.")
    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Target 'churn' must contain only Yes/No values.")
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
    return df, X, y

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    results, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        results.append(metrics)
        fitted[name] = pipe
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k,v in metrics.items() if k != "model"))
    metrics_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    best_name = str(metrics_df.iloc[0]["model"])
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"model_name": best_name, "target": TARGET,
                   "features": X.columns.tolist(), "dataset_rows": int(len(df)),
                   "synthetic_demo_data": True}, f, indent=2)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    (REPORT_DIR / "classification_report.txt").write_text(
        f"Selected model: {best_name}\n\n" + classification_report(y_test, y_pred, target_names=["No churn", "Churn"]),
        encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No churn", "Churn"], ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("ROC Curves — Test Set")
    fig.tight_layout(); fig.savefig(FIG_DIR / "roc_curves.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = metrics_df.set_index("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    plot_df.plot(kind="bar", ax=ax, ylim=(0, 1), rot=15)
    ax.set_ylabel("Score"); ax.set_title("Model Comparison (test set)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "model_comparison.png", dpi=160); plt.close(fig)
    # Feature importance when supported by the fitted estimator.
    model_step = best_model.named_steps["model"]
    prep = best_model.named_steps["preprocess"]
    try:
        names = prep.get_feature_names_out()
        if hasattr(model_step, "feature_importances_"):
            vals = model_step.feature_importances_
        elif hasattr(model_step, "coef_"):
            vals = np.abs(model_step.coef_[0])
        else:
            vals = None
        if vals is not None:
            imp = pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance", ascending=False).head(15)
            imp.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
            fig, ax = plt.subplots(figsize=(9, 6))
            imp.sort_values("importance").plot.barh(x="feature", y="importance", ax=ax, legend=False)
            ax.set_title(f"Top Predictive Features — {best_name}")
            fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png", dpi=160); plt.close(fig)
    except Exception as e:
        print(f"Feature importance plot skipped: {e}")
    print(f"\nSelected model: {best_name}")
    print(f"Saved model to {MODEL_DIR / 'best_model.joblib'}")
    print(f"Saved reports to {REPORT_DIR}")

if __name__ == "__main__":
    main()
