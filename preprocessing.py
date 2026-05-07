"""
preprocessing.py
----------------
Loads the student CSV, scales features to [0, 1] (required for angle-based
quantum encoding), reduces dimensionality to 2 principal components (one
qubit per component), and produces train / test splits.

Exported symbols used by other modules
  X_train, X_test, y_train, y_test  – ready-to-use numpy arrays
  scaler                             – fitted MinMaxScaler
  pca                                – fitted PCA(n_components=2)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# ── Load raw data ─────────────────────────────────────────────────────────────
df = pd.read_csv("students.csv")

X_raw = df[["study_hours", "attendance", "sleep_hours"]].values
y     = df["result"].values.astype(int)

# ── Step 1 – MinMax scaling to [0, 1] ────────────────────────────────────────
# Quantum feature maps (e.g. ZZFeatureMap) use the raw values as rotation
# angles; keeping them in [0, 1] → angles in [0, π] avoids wrap-around issues.
scaler   = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_raw)

# ── Step 2 – PCA: 3 features → 2 principal components ───────────────────────
# The VQC uses 2 qubits, so we need exactly 2 input features.
# PCA retains maximum variance while matching the circuit width.
pca       = PCA(n_components=2, random_state=42)
X_reduced = pca.fit_transform(X_scaled)

# ── Step 3 – Re-scale PCA output to [0, 1] ───────────────────────────────────
# PCA components can be negative; a second MinMax pass keeps angles positive.
pca_scaler   = MinMaxScaler(feature_range=(0, 1))
X_final      = pca_scaler.fit_transform(X_reduced)

# ── Step 4 – Train / test split (80 / 20) ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)


def preprocess_single(study_hours: float,
                      attendance:  float,
                      sleep_hours: float) -> np.ndarray:
    """
    Apply the SAME pipeline to a single new student record.
    Returns a (1, 2) array ready for VQC.predict().
    """
    raw     = np.array([[study_hours, attendance, sleep_hours]])
    scaled  = scaler.transform(raw)
    reduced = pca.transform(scaled)
    final   = pca_scaler.transform(reduced)
    return final


if __name__ == "__main__":
    print("Preprocessing summary")
    print(f"  Total samples : {len(y)}")
    print(f"  X_train shape : {X_train.shape}")
    print(f"  X_test  shape : {X_test.shape}")
    print(f"  Feature range after PCA rescale:")
    print(f"    min={X_final.min():.4f}  max={X_final.max():.4f}")
    print(f"  Class balance (train) – PASS:{y_train.sum()}  FAIL:{(y_train==0).sum()}")
