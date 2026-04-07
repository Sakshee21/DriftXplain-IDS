from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    out_csv = root / "tests" / "demo_inputs.csv"

    rng = np.random.default_rng(42)

    model = joblib.load(models_dir / "rf_final.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    features = list(scaler.feature_names_in_)

    x = pd.DataFrame(10 ** rng.uniform(-3, 5, size=(8000, len(features))), columns=features)
    x_scaled = pd.DataFrame(scaler.transform(x), columns=features)
    probs = model.predict_proba(x_scaled)[:, 1]

    targets = [0.05, 0.40, 0.90]
    labels = ["likely_normal", "borderline_mixed", "likely_attack"]
    pick_idx = [int(np.argmin(np.abs(probs - t))) for t in targets]

    out = x.iloc[pick_idx].copy()
    out.insert(0, "case", labels)
    out.to_csv(out_csv, index=False)

    print(f"Saved {out_csv}")
    print("Picked probabilities:", probs[pick_idx])


if __name__ == "__main__":
    main()
