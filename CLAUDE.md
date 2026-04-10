# NN Pipeline — Project Brief

## What this is
A 10-stage ML preprocessing + classification pipeline built in **Plotly Dash** (Python).
Input: raw survey data (CSV/Excel) + questionnaire (.docx/.pdf/.xlsx/.txt).
Output: trained PyTorch neural network for multiclass classification.

## Repo
- **GitHub**: https://github.com/azat-shatru/NN_classification
- **Active branch**: `dash-app`
- **Main branch**: `main`

## How to run
```bash
.venv/Scripts/python dash_app/app.py
# → http://127.0.0.1:8050
```

## Project structure
```
dash_app/
  app.py              # Dash entry point, routing, sidebar, layout
  server_store.py     # In-memory store for DataFrames/models (avoids browser serialisation)
  pages/
    var_mapping.py    # Pre-stage: upload data + QNR, interactive tile mapping
    stage0_audit.py   # Data audit, column editor, drop/type assignment
    stage1_missing.py # Missing value imputation
    stage2_outliers.py
    stage2b_viz.py    # Visualisation
    stage3_encoding.py
    stage4_scaling.py
    stage5_correlation.py
    stage6_rf.py      # Random Forest feature importance
    stage7_factor.py  # Factor analysis
    stage8_combtest.py
    stage9_nn.py      # PyTorch NN training
  utils/
    qnr_parser.py     # Parses docx/pdf/xlsx/txt questionnaire → list of question dicts
    col_mapper.py     # Groups dataset columns by prefix, suggests var types
  assets/style.css    # All custom CSS
update_log.py         # Run this after every session to log changes to NN_Design_Tracker.xlsx
NN_Design_Tracker.xlsx  # Session log (Sheet: "Session Log"), next session = 17
```

## Architecture decisions
- **server_store.py** holds all DataFrames in process memory — do NOT put large data in `dcc.Store` (browser JSON)
- **app-state** (`dcc.Store`, session) holds lightweight pipeline flags (booleans, column name lists)
- **vm-state** (`dcc.Store`, memory) holds Variable Mapping mutations (deleted cols, type overrides, reassignments)
- Sidebar collapse state in `dcc.Store(id="sidebar-collapsed", storage_type="session")`
- All pages use `suppress_callback_exceptions=True` (set in app.py)
- `debug=False` — keep it off to suppress Plotly cloud prompt

## Key conventions
- Every `html.A` inside `dcc.Upload` must be `html.Span` (html.A with no href causes page refresh)
- Pattern-matching callbacks use `ctx.triggered_id` to identify which component fired
- `allow_duplicate=True` required when multiple callbacks write to the same Output
- After every coding session: update `update_log.py` and run it → logs to NN_Design_Tracker.xlsx

## Variable Mapping page — key details
- `_render_tile()` signature: `(col, df, options_str, scale_str, current_type, current_q_code, all_q_codes, is_possibly_related, is_extra)`
- Options shown inline on each tile as pill chips (max 6 shown, "+N more" for remainder)
- Scale questions show `Scale: 1 = "low" → N = "high"` instead of chips
- `qnr_parser._parse_docx` reads docx in document order (paragraphs + tables interleaved) to preserve question→options linkage; handles single-cell, value|label, and multi-cell table rows

## Current state (last updated 2026-04-10, session 16)
- Variable Mapping page: accordion tiles with inline options per tile, hover preview, delete/move/reassign
- qnr_parser: fixed docx parsing to read in document order (options in tables now correctly linked to questions)
- Stage 0: upload fix (html.A → html.Span), column editor drop fix
- Sidebar collapse toggle implemented
- Stages 1–9: inherited from original Streamlit rewrite, not yet re-tested end-to-end

## Open tasks
- Test full pipeline end-to-end with MyRawdata.xlsx + survey.docx
- A7 value=8 (NEE) custom missing value handling (Stage 1)
- Re-zip and upload to Google Drive
