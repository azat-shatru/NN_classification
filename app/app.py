"""
Main entry point — run with:
    streamlit run app/app.py
"""
import streamlit as st
from utils.state import init

st.set_page_config(
    page_title="NN Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

init()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🧠 NN Pipeline")
st.sidebar.caption("Multiclass Classification · PyTorch")
st.sidebar.divider()

PAGES = {
    "Stage 0 · Data Audit":            "0_audit",
    "Stage 1 · Missing Values":        "1_missing",
    "Stage 2 · Outlier Treatment":     "2_outliers",
    "Stage 2.5 · Visualisation":       "2b_viz",
    "Stage 3 · Encoding":              "3_encoding",
    "Stage 4 · Scaling":               "4_scaling",
    "Stage 5 · Correlation Filter":    "5_correlation",
    "Stage 6 · RF Importance":         "6_rf",
    "Stage 7 · Factor Analysis":       "7_factor",
    "Stage 8 · Combination Testing":   "8_combtest",
    "Stage 9 · Neural Network":        "9_nn",
}

# Status dots — green if stage completed
STATUS = {
    "0_audit":       "df" in st.session_state and st.session_state["df"] is not None,
    "1_missing":     st.session_state.get("df_imputed", False),
    "2_outliers":    st.session_state.get("df_outliers_done", False),
    "2b_viz":        st.session_state.get("codebook") is not None,
    "3_encoding":    st.session_state.get("df_encoded", False),
    "4_scaling":     st.session_state.get("df_scaled", False),
    "5_correlation": st.session_state.get("corr_done", False),
    "6_rf":          st.session_state.get("rf_done", False),
    "7_factor":      st.session_state.get("factor_done", False),
    "8_combtest":    st.session_state.get("selected_features") is not None,
    "9_nn":          st.session_state.get("nn_trained", False),
}

selected_label = st.sidebar.radio(
    "Navigate",
    list(PAGES.keys()),
    format_func=lambda x: ("✅ " if STATUS.get(PAGES[x]) else "⬜ ") + x
)

selected_key = PAGES[selected_label]

st.sidebar.divider()
st.sidebar.caption("Progress")
done = sum(STATUS.values())
st.sidebar.progress(done / len(STATUS))
st.sidebar.caption(f"{done} / {len(STATUS)} stages complete")

# ── Route to selected stage ───────────────────────────────────────────────────
if selected_key == "0_audit":
    from stages.stage0_audit import show
    show()

elif selected_key == "1_missing":
    from stages.stage1_missing import show
    show()

elif selected_key == "2_outliers":
    from stages.stage2_outliers import show
    show()

elif selected_key == "2b_viz":
    from stages.stage2b_viz import show
    show()

elif selected_key == "3_encoding":
    from stages.stage3_encoding import show
    show()

elif selected_key == "4_scaling":
    from stages.stage4_scaling import show
    show()

elif selected_key == "5_correlation":
    from stages.stage5_correlation import show
    show()

elif selected_key == "6_rf":
    from stages.stage6_rf import show
    show()

elif selected_key == "7_factor":
    from stages.stage7_factor import show
    show()

elif selected_key == "8_combtest":
    from stages.stage8_combtest import show
    show()

elif selected_key == "9_nn":
    from stages.stage9_nn import show
    show()
