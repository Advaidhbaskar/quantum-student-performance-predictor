"""
app.py
------
Streamlit UI for the Quantum Student Grade Predictor.

Run with:
    streamlit run app.py

Features
  • Train both Quantum VQC and Classical LogReg models from the UI
  • Live training-loss progress bar (quantum model)
  • Single-student PASS / FAIL prediction from both models
  • Side-by-side accuracy comparison chart
  • Training-loss curve and confusion matrix display
"""

import os
import pickle
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Grade Predictor",
    page_icon="⚛️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global typography ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Header strip ── */
.hero {
    background: linear-gradient(135deg, #0d0d2b 0%, #1a1a4e 50%, #0d1f3c 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid #2a2a6e;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(67,97,238,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    color: #e0e7ff;
    font-size: 2.2rem;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #94a3c8;
    margin: 0;
    font-size: 1.0rem;
    font-weight: 300;
}
.hero .badge {
    display: inline-block;
    background: rgba(67,97,238,0.25);
    border: 1px solid #4361ee;
    color: #818cf8;
    border-radius: 99px;
    padding: 2px 14px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.7rem;
}

/* ── Metric cards ── */
.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-card .label {
    color: #6b7280;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e0e7ff;
}

/* ── Result banners ── */
.result-pass {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #16a34a;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-family: 'Space Mono', monospace;
}
.result-pass h2 { color: #4ade80; font-size: 2rem; margin: 0; }
.result-fail {
    background: linear-gradient(135deg, #1c0a0a, #450a0a);
    border: 1px solid #dc2626;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-family: 'Space Mono', monospace;
}
.result-fail h2 { color: #f87171; font-size: 2rem; margin: 0; }

/* ── Section header ── */
.section-header {
    font-family: 'Space Mono', monospace;
    color: #818cf8;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.6rem 0;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 0.4rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0b0f1a !important;
    border-right: 1px solid #1f2937 !important;
}
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper – safe model loading
# ═══════════════════════════════════════════════════════════════════════════════

def _model_exists(path: str) -> bool:
    return os.path.exists(path)


@st.cache_resource(show_spinner=False)
def _load_vqc():
    from quantum_model import load_model as _qload
    return _qload()


@st.cache_resource(show_spinner=False)
def _load_classical():
    with open("classical_model.pkl", "rb") as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Hero header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="badge">⚛ Quantum Machine Learning</div>
  <h1>Student Grade Predictor</h1>
  <p>Variational Quantum Classifier (VQC) with ZZFeatureMap + RealAmplitudes ansatz</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar – student input
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎓 Student Profile")
    st.caption("Enter the student's details below.")

    study_hours = st.slider("📚 Study Hours / Day", 1.0, 10.0, 5.5, 0.5,
                            help="Average hours spent studying each day.")
    attendance  = st.slider("📅 Attendance (%)", 50.0, 100.0, 75.0, 1.0,
                            help="Percentage of classes attended.")
    sleep_hours = st.slider("😴 Sleep Hours / Night", 4.0, 9.0, 7.0, 0.5,
                            help="Average nightly sleep hours.")

    st.markdown("---")
    predict_btn = st.button("⚛️  Predict with VQC", use_container_width=True,
                             type="primary")
    compare_btn = st.button("🔬  Compare Classical vs Quantum",
                             use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚙️ Model Controls")
    train_btn = st.button("🚀  Train / Re-train Models", use_container_width=True)

    with st.expander("Circuit parameters"):
        st.markdown("""
| Parameter | Value |
|---|---|
| Qubits | 2 |
| Feature map | ZZFeatureMap (reps=2) |
| Ansatz | RealAmplitudes (reps=3) |
| Optimizer | COBYLA |
| Max iterations | 150 |
| Sampler | StatevectorSampler |
""")

# ═══════════════════════════════════════════════════════════════════════════════
# Training section
# ═══════════════════════════════════════════════════════════════════════════════

if train_btn:
    st.markdown('<div class="section-header">Model Training</div>',
                unsafe_allow_html=True)

    # ── Classical ─────────────────────────────────────────────────────────────
    with st.spinner("Training Classical Logistic Regression …"):
        from classical_model import train_classical
        c_result = train_classical()
    st.success(f"✅ Classical model trained  –  "
               f"Accuracy: **{c_result['accuracy']*100:.1f} %**")

    # ── Quantum ───────────────────────────────────────────────────────────────
    st.info("⚛️  Training Quantum VQC …  this may take 1–3 minutes.")

    loss_placeholder = st.empty()
    progress_bar     = st.progress(0)
    from quantum_model import MAX_ITER

    def _ui_callback(iteration, loss):
        frac = min(iteration / MAX_ITER, 1.0)
        progress_bar.progress(frac)
        loss_placeholder.caption(
            f"Iteration {iteration}/{MAX_ITER}  |  Loss: {loss:.4f}"
        )

    from quantum_model import train_and_evaluate
    q_result = train_and_evaluate(progress_callback=_ui_callback)

    progress_bar.progress(1.0)
    loss_placeholder.empty()
    st.success(f"✅ Quantum VQC trained  –  "
               f"Accuracy: **{q_result['accuracy']*100:.1f} %**")

    # Invalidate cached models so fresh ones are loaded
    _load_vqc.clear()
    _load_classical.clear()

    st.balloons()


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction section
# ═══════════════════════════════════════════════════════════════════════════════

if predict_btn:
    st.markdown('<div class="section-header">Quantum Prediction</div>',
                unsafe_allow_html=True)

    if not _model_exists("vqc_weights.npy"):
        st.error("No trained VQC found.  Click **Train / Re-train Models** first.")
    else:
        from preprocessing import preprocess_single
        features = preprocess_single(study_hours, attendance, sleep_hours)

        with st.spinner("Running quantum circuit …"):
            vqc  = _load_vqc()
            pred = int(np.asarray(vqc.predict(features)).flat[0])

        label = "PASS" if pred == 1 else "FAIL"
        css   = "result-pass" if pred == 1 else "result-fail"
        icon  = "🎉" if pred == 1 else "📛"

        st.markdown(f"""
        <div class="{css}">
          <p style="color:#94a3b8;font-size:0.85rem;margin:0 0 0.3rem 0;">
            VQC Prediction for  Study={study_hours}h  |  Attendance={attendance}%  |  Sleep={sleep_hours}h
          </p>
          <h2>{icon}  {label}</h2>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison section
# ═══════════════════════════════════════════════════════════════════════════════

if compare_btn:
    st.markdown('<div class="section-header">Quantum vs Classical Comparison</div>',
                unsafe_allow_html=True)

    missing = []
    if not _model_exists("vqc_weights.npy"):       missing.append("VQC")
    if not _model_exists("classical_model.pkl"): missing.append("Classical")

    if missing:
        st.error(f"Missing trained model(s): {', '.join(missing)}.  "
                 "Click **Train / Re-train Models** first.")
    else:
        from preprocessing import preprocess_single, X_test, y_test

        features    = preprocess_single(study_hours, attendance, sleep_hours)
        vqc         = _load_vqc()
        classical   = _load_classical()

        q_pred_user = int(np.asarray(vqc.predict(features)).flat[0])
        c_pred_user = int(classical.predict(features)[0])

        q_test_pred = vqc.predict(X_test)
        c_test_pred = classical.predict(X_test)
        q_acc       = accuracy_score(y_test, q_test_pred)
        c_acc       = accuracy_score(y_test, c_test_pred)

        # ── Metric cards ──────────────────────────────────────────────────────
        col1, col2 = st.columns(2)

        def _result_html(pred: int, label: str) -> str:
            colour = "#4ade80" if pred == 1 else "#f87171"
            text   = "PASS" if pred == 1 else "FAIL"
            return f"""
            <div class="metric-card">
              <div class="label">{label} Prediction</div>
              <div class="value" style="color:{colour}">{text}</div>
            </div>"""

        with col1:
            st.markdown(_result_html(q_pred_user, "⚛️ Quantum VQC"),
                        unsafe_allow_html=True)
        with col2:
            st.markdown(_result_html(c_pred_user, "🖥️ Classical LogReg"),
                        unsafe_allow_html=True)

        # ── Bar chart ─────────────────────────────────────────────────────────
        st.markdown("#### Test-set Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        bars = ax.bar(
            ["Classical\nLogistic Regression", "Quantum\nVQC"],
            [c_acc * 100, q_acc * 100],
            color=["#64748b", "#4361EE"],
            width=0.45,
            edgecolor="none",
            zorder=3,
        )
        for bar, val in zip(bars, [c_acc, q_acc]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val*100:.1f} %",
                    ha="center", va="bottom",
                    fontweight="bold", fontsize=12, color="#e0e7ff")

        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)", color="#94a3b8")
        ax.set_title("Quantum vs Classical – Test Accuracy",
                     fontsize=12, color="#e0e7ff", pad=10)
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.grid(True, color="#1f2937", zorder=0)
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")
        st.pyplot(fig)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisations (always shown if plots exist)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Training Visualisations</div>',
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    if os.path.exists("accuracy_plot.png"):
        st.image("accuracy_plot.png",
                 caption="VQC Training-Loss Curve",
                 use_container_width=True)
    else:
        st.info("Train the model to see the loss curve.")

with col_b:
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png",
                 caption="VQC Confusion Matrix",
                 use_container_width=True)
    else:
        st.info("Train the model to see the confusion matrix.")


# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<hr style="border:none;border-top:1px solid #1f2937;margin:2rem 0 1rem 0;">
<p style="text-align:center;color:#374151;font-size:0.78rem;font-family:'Space Mono',monospace;">
  ⚛️  Quantum Student Grade Predictor  ·  ZZFeatureMap + RealAmplitudes  ·  Qiskit 2.x
</p>
""", unsafe_allow_html=True)
