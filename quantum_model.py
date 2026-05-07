"""
quantum_model.py
----------------
VQC pipeline: ZZFeatureMap -> RealAmplitudes -> COBYLA -> StatevectorSampler

Persistence fix:
  - Cannot pickle VQC (internal lambdas)
  - Cannot set vqc.weights (read-only property in qiskit-ml >= 0.7)
  Solution: save weights with np.save, reload by passing as initial_point
            with maxiter=0 so the optimizer does zero steps — weights stay exact.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

from preprocessing import X_train, X_test, y_train, y_test

warnings.filterwarnings("ignore")

WEIGHTS_PATH   = "vqc_weights.npy"
ACCURACY_PLOT  = "accuracy_plot.png"
CONFUSION_PLOT = "confusion_matrix.png"

NUM_QUBITS       = 2
FEATURE_MAP_REPS = 2
ANSATZ_REPS      = 3
MAX_ITER         = 150


def _make_vqc(callback=None, initial_point=None, maxiter=None) -> VQC:
    """Construct a VQC with the project architecture."""
    feature_map = zz_feature_map(feature_dimension=NUM_QUBITS, reps=FEATURE_MAP_REPS)
    ansatz      = real_amplitudes(num_qubits=NUM_QUBITS, reps=ANSATZ_REPS)
    optimizer   = COBYLA(maxiter=maxiter if maxiter is not None else MAX_ITER)
    sampler     = StatevectorSampler()
    return VQC(
        feature_map   = feature_map,
        ansatz        = ansatz,
        optimizer     = optimizer,
        sampler       = sampler,
        callback      = callback,
        initial_point = initial_point,
    )


def train_and_evaluate(progress_callback=None) -> dict:
    """Train VQC, save weights array, save plots, return metrics."""
    loss_history = []

    def _cb(weights, loss):
        loss_history.append(float(loss))
        if progress_callback:
            progress_callback(len(loss_history), float(loss))

    vqc = _make_vqc(callback=_cb)
    print(f"Training VQC — {NUM_QUBITS} qubits, reps={ANSATZ_REPS}, maxiter={MAX_ITER}")
    vqc.fit(X_train, y_train)

    y_pred   = vqc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Quantum VQC Accuracy: {accuracy * 100:.2f} %")

    # Save only the weights numpy array — avoids pickle-lambda AND read-only issues
    np.save(WEIGHTS_PATH, vqc.weights)
    print(f"Weights saved -> {WEIGHTS_PATH}  shape={vqc.weights.shape}")

    _save_loss_plot(loss_history, accuracy)
    _save_confusion_matrix(y_test, y_pred)

    return {"accuracy": accuracy, "y_pred": y_pred, "loss_history": loss_history}


def load_model() -> VQC:
    """
    Restore a trained VQC without re-training.

    Strategy: pass saved weights as initial_point AND set maxiter=0 so
    COBYLA does zero optimization steps — the model predicts with the
    exact saved weights.
    """
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"No weights at '{WEIGHTS_PATH}'. Train the model first."
        )

    saved_weights = np.load(WEIGHTS_PATH)

    # Build VQC with saved weights as starting point and 0 optimizer steps
    vqc = _make_vqc(initial_point=saved_weights, maxiter=0)

    # fit() with maxiter=0 initialises all internal state (output_shape,
    # interpret function, etc.) but does not move the weights at all.
    vqc.fit(X_train[:8], y_train[:8])

    return vqc


def predict_student(study_hours: float, attendance: float, sleep_hours: float) -> str:
    """Single-student quantum prediction."""
    from preprocessing import preprocess_single
    vqc      = load_model()
    features = preprocess_single(study_hours, attendance, sleep_hours)
    pred     = vqc.predict(features)
    return "PASS" if int(pred[0]) == 1 else "FAIL"


def _save_loss_plot(loss_history, accuracy):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history, color="#4361EE", linewidth=2, label="Training Loss")
    ax.set_xlabel("Optimizer Iteration", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(f"VQC Training Loss  |  Test Accuracy: {accuracy*100:.1f} %",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ACCURACY_PLOT, dpi=150); plt.close(fig)


def _save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["FAIL", "PASS"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("VQC Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(CONFUSION_PLOT, dpi=150); plt.close(fig)


if __name__ == "__main__":
    r = train_and_evaluate()
    print(f"Done. Accuracy = {r['accuracy']*100:.2f} %")
