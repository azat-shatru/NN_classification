"""
Central session-state keys and initialisation helper.
Every stage reads/writes through these keys so nothing gets lost
when the user navigates between pages.
"""
import streamlit as st

KEYS = {
    "raw_df":          None,   # original uploaded dataframe
    "df":              None,   # working dataframe (mutated per stage)
    "target_col":      None,   # name of the target / label column
    "train_idx":       None,   # row indices for train split
    "test_idx":        None,   # row indices for test split
    "X_train":         None,   # feature matrix — train
    "X_test":          None,   # feature matrix — test
    "y_train":         None,   # labels — train
    "y_test":          None,   # labels — test
    "codebook":        None,   # list[dict] — variable metadata + label maps
    "selected_features": None, # list[str] — final features after Stage 8
    "numeric_cols":    None,
    "categorical_cols": None,
    "ordinal_cols":    None,
    # Stage flags
    "df_imputed":      False,
    "df_outliers_done": False,
    "df_encoded":      False,
    "df_scaled":       False,
    "corr_done":       False,
    "rf_done":         False,
    "factor_done":     False,
    "nn_trained":      False,
    # Stage 0 artefacts
    "dropped_cols_audit": [],
    # Stage artefacts
    "parsed_questions": None,
    "rf_importances":  None,
    "fa_loadings":     None,
    "shap_df":         None,
    "scalers_used":    None,
    "enc_strategies":  None,
    "ord_orders":      None,
    "comb_results":    None,
    "fw_history":      None,
    "bw_history":      None,
}

def init():
    for k, v in KEYS.items():
        if k not in st.session_state:
            st.session_state[k] = v
