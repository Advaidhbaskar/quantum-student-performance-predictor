"""
classical_model.py
------------------
Trains a classical Logistic Regression baseline on the same preprocessed
features used by the VQC.  This lets us compare quantum vs classical
performance side-by-side in the Streamlit UI.

The fitted model is pickled alongside the quantum model so the UI can
display both predictions simultaneously.
"""

import pickle
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from preprocessing import X_train, X_test, y_train, y_test

warnings.filterwarnings("ignore")

CLASSICAL_MODEL_PATH = "classical_model.pkl"


def train_classical() -> dict:
    """
    Fit a Logistic Regression classifier and persist it.

    Returns
    -------
    dict with keys: accuracy, y_pred, model
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    with open(CLASSICAL_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Classical (LogReg) Accuracy : {accuracy * 100:.2f} %")
    print(f"Model saved → {CLASSICAL_MODEL_PATH}")
    return {"accuracy": accuracy, "y_pred": y_pred, "model": model}


def load_classical_model():
    """Load the persisted classical model."""
    import os
    if not os.path.exists(CLASSICAL_MODEL_PATH):
        raise FileNotFoundError(
            f"'{CLASSICAL_MODEL_PATH}' not found.  Train the model first."
        )
    with open(CLASSICAL_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_classical(study_hours: float,
                      attendance:  float,
                      sleep_hours: float) -> str:
    """Single-student classical prediction (for UI comparison)."""
    from preprocessing import preprocess_single
    model    = load_classical_model()
    features = preprocess_single(study_hours, attendance, sleep_hours)
    pred     = model.predict(features)
    return "PASS" if int(pred[0]) == 1 else "FAIL"


if __name__ == "__main__":
    results = train_classical()
    print(f"\nDone.  Classical Accuracy = {results['accuracy']*100:.2f} %")
