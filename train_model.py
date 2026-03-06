"""Train and persist the Employee Access classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = Path("amazon_employee_access_train.csv")
MODEL_PATH = Path("artifacts/model.joblib")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "ACTION" not in df.columns:
        raise ValueError("Training data must include ACTION target column")

    X = df.drop(columns=["ACTION"])
    y = df["ACTION"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight={0: 2, 1: 1},
        random_state=123,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
