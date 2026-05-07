# ⚛️ Quantum Student Grade Predictor

A **Variational Quantum Classifier (VQC)** that predicts student PASS / FAIL
outcomes from three inputs — study hours, attendance, and sleep hours — with a
clean Streamlit UI and side-by-side comparison against a classical baseline.

---

## Quantum Circuit Design

```
Input (2 PCA features)
        │
        ▼
┌──────────────────────────────────────────┐
│  ZZFeatureMap  (reps=2)                  │
│  H ─ P(2x₀) ─ ZZ(x₀·x₁) ─ H ─ P(2x₀) │  ← non-linear quantum encoding
│  H ─ P(2x₁) ─ ZZ(x₀·x₁) ─ H ─ P(2x₁) │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  RealAmplitudes ansatz  (reps=3)         │
│  Ry(θ₀) ─■─ Ry(θ₂) ─■─ Ry(θ₄)        │  ← trainable parameters θ
│  Ry(θ₁) ─X─ Ry(θ₃) ─X─ Ry(θ₅)        │
└──────────────────────────────────────────┘
        │
        ▼
  Measure → COBYLA optimises θ to minimise cross-entropy loss
        │
        ▼
   PASS / FAIL
```

---

## Project Structure

```
student_vqc_predictor/
├── app.py              ← Streamlit UI (run this)
├── quantum_model.py    ← VQC build, train, predict, save plots
├── classical_model.py  ← Logistic Regression baseline
├── preprocessing.py    ← Scale → PCA → rescale → train/test split
├── dataset.py          ← Generates students.csv
├── students.csv        ← Pre-generated dataset (200 samples)
├── requirements.txt
└── README.md
```

After training the following artefacts are created automatically:

| File | Description |
|---|---|
| `vqc_model.pkl` | Trained VQC (pickle) |
| `classical_model.pkl` | Trained LogReg (pickle) |
| `accuracy_plot.png` | VQC training-loss curve |
| `confusion_matrix.png` | VQC confusion matrix |

---

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Python 3.9 – 3.11** is recommended.  
> Qiskit 2.x is not yet fully supported on Python 3.12.

---

## Running the Project

### Option A — Streamlit UI (recommended)

```bash
cd student_vqc_predictor
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

**Workflow inside the UI:**

1. Click **"🚀 Train / Re-train Models"** in the sidebar.  
   (Quantum training takes ~1–3 min; a live progress bar is shown.)
2. Adjust the student sliders (study hours, attendance, sleep).
3. Click **"⚛️ Predict with VQC"** for a quantum PASS/FAIL result.
4. Click **"🔬 Compare Classical vs Quantum"** for a side-by-side comparison.

### Option B — Command Line (headless)

```bash
# Train quantum model (saves model + plots)
python quantum_model.py

# Train classical baseline
python classical_model.py
```

---

## How It Works

| Step | File | What happens |
|---|---|---|
| Data generation | `dataset.py` | Synthetic 200-student CSV with weighted PASS/FAIL rule |
| Preprocessing | `preprocessing.py` | MinMax scale → PCA(2) → rescale → stratified split |
| Quantum model | `quantum_model.py` | ZZFeatureMap + RealAmplitudes + COBYLA + StatevectorSampler |
| Classical model | `classical_model.py` | LogisticRegression baseline |
| UI | `app.py` | Streamlit app with training, prediction, and visualisation |

---

## Notes

- **StatevectorSampler** is used for exact (noiseless) simulation — training is
  deterministic and reproducible.
- **COBYLA** (gradient-free) is chosen because quantum circuits do not provide
  analytical gradients through the shot-based sampling interface.
- The 3-feature dataset is reduced to **2 PCA components** so the quantum circuit
  width exactly matches the number of input qubits.
