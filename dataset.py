"""
dataset.py
----------
Generates a synthetic student dataset with three features:
  - study_hours   : hours spent studying per day  (1–10)
  - attendance    : class attendance percentage    (50–100)
  - sleep_hours   : hours of sleep per night       (4–9)

The PASS/FAIL label is determined by a weighted score threshold so the
dataset has a meaningful (non-trivial) decision boundary that challenges
both classical and quantum classifiers.
"""

import numpy as np
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
N = 200

# ── Raw feature generation ────────────────────────────────────────────────────
study_hours = np.random.uniform(1, 10, N)
attendance  = np.random.uniform(50, 100, N)
sleep_hours = np.random.uniform(4, 9, N)

# ── Normalise to [0, 1] for scoring (keeps weights interpretable) ─────────────
study_norm      = study_hours / 10.0
attendance_norm = attendance  / 100.0
sleep_norm      = sleep_hours / 9.0

# ── Weighted rule (study is the dominant predictor) ───────────────────────────
score = (
    0.55 * study_norm +       # most important factor
    0.20 * attendance_norm +  # moderate importance
    0.25 * sleep_norm         # health / retention factor
)

# Students whose weighted score exceeds the threshold PASS
THRESHOLD = 0.55
result = (score > THRESHOLD).astype(int)

# ── Persist ───────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance":  attendance,
    "sleep_hours": sleep_hours,
    "result":      result,
})
df.to_csv("students.csv", index=False)

# ── Debug summary ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pass_count = result.sum()
    print(f"Dataset created  →  {N} students")
    print(f"  PASS : {pass_count}  ({pass_count/N*100:.1f} %)")
    print(f"  FAIL : {N - pass_count}  ({(N-pass_count)/N*100:.1f} %)")
    print(f"Saved to students.csv")
