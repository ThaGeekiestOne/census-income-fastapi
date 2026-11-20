"""Train the model, save artifacts, and write slice metrics."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics, compute_slice_metrics, inference,
    save_artifact, train_model,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def main():
    data = pd.read_csv(ROOT / "data" / "census.csv")
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary",
        training=False, encoder=encoder, lb=lb,
    )

    model = train_model(X_train, y_train)
    precision, recall, f1 = compute_model_metrics(
        y_test, inference(model, X_test))
    print(f"Test precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    save_artifact(model, MODEL_DIR / "model.joblib")
    save_artifact(encoder, MODEL_DIR / "encoder.joblib")
    save_artifact(lb, MODEL_DIR / "lb.joblib")

    # Slice metrics for every categorical feature -> slice_output.txt
    with open(ROOT / "slice_output.txt", "w") as f:
        for feature in CAT_FEATURES:
            result = compute_slice_metrics(
                test, feature, model, encoder, lb, CAT_FEATURES)
            f.write(f"=== Slices for: {feature} ===\n")
            f.write(result.to_string(index=False) + "\n\n")


if __name__ == "__main__":
    main()
