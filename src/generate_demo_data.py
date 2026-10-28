"""Generate a reproducible synthetic telecom churn dataset for an offline demo."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "customer_churn_demo.csv"

def main(n_rows: int = 3000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n_rows)
    monthly_charges = np.round(rng.uniform(20, 125, n_rows), 2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_rows, p=[0.35, 0.48, 0.17])
    tech_support = rng.choice(["Yes", "No", "No internet service"], n_rows, p=[0.28, 0.55, 0.17])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_rows)
    paperless = rng.choice(["Yes", "No"], n_rows, p=[0.65, 0.35])
    senior = rng.choice([0, 1], n_rows, p=[0.84, 0.16])
    support_penalty = (tech_support == "No") * 0.55
    contract_penalty = (contract == "Month-to-month") * 1.15 + (contract == "One year") * 0.25
    fiber_penalty = (internet == "Fiber optic") * 0.25
    payment_penalty = (payment == "Electronic check") * 0.40
    linear = -2.0 + contract_penalty + support_penalty + fiber_penalty + payment_penalty
    linear += (monthly_charges - 70) / 65 * 0.65 + senior * 0.25 - tenure / 40 * 1.5
    linear += (paperless == "Yes") * 0.10
    probability = 1 / (1 + np.exp(-linear))
    churn = rng.binomial(1, probability)
    total_charges = np.maximum(0, tenure * monthly_charges + rng.normal(0, 60, n_rows)).round(2)
    df = pd.DataFrame({
        "customer_id": [f"DEMO{i:05d}" for i in range(n_rows)],
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "internet_service": internet,
        "tech_support": tech_support,
        "payment_method": payment,
        "paperless_billing": paperless,
        "senior_citizen": senior,
        "churn": np.where(churn == 1, "Yes", "No"),
    })
    # A small number of missing values makes the preprocessing step meaningful.
    for col in ["total_charges", "tech_support"]:
        ix = rng.choice(n_rows, max(1, int(n_rows * 0.01)), replace=False)
        df.loc[ix, col] = np.nan
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Generated {len(df):,} synthetic demo rows: {OUT}")
    print("NOTE: This is synthetic data for demonstrating the pipeline, not real customer data.")

if __name__ == "__main__":
    main()
