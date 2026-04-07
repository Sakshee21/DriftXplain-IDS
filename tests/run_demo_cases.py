from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    test_csv = root / "tests" / "demo_inputs.csv"

    model = joblib.load(models_dir / "rf_final.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    features = list(scaler.feature_names_in_)

    df_cases = pd.read_csv(test_csv)

    missing = [f for f in features if f not in df_cases.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    x = df_cases[features].copy()
    x_scaled = scaler.transform(x)
    x_scaled_df = pd.DataFrame(x_scaled, columns=features)
    probs = model.predict_proba(x_scaled_df)[:, 1]

    thresholds = [0.30, 0.40, 0.50]
    out = pd.DataFrame(
        {
            "case": df_cases["case"],
            "attack_probability": probs,
        }
    )

    for t in thresholds:
        out[f"pred_at_{t:.2f}"] = (out["attack_probability"] >= t).astype(int)

    out = out.sort_values("attack_probability", ascending=False).reset_index(drop=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print("Demo predictions (higher probability means more attack-like):")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)))


if __name__ == "__main__":
    main()
