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
- Layout: accordion card per question; body is a 4-col table (Variable | Options | Type | Question)
- Options appear as draggable chips in col 2; JS drag-and-drop (assets/dragdrop.js) via document-level event delegation
- Auto-assignment logic: 1 col → all opts; multi q_type + suffix → indexed opt; grid → all opts per col; scale/numeric → no chips
- Unassigned options pool row shown at bottom of each table when options remain unmatched
- "Save changes" button → clientside_callback reads DOM chip positions → vm-drag-assignments store → handle_save server callback
- `vm-state` uses `storage_type="session"` so assignments persist across page navigation
- `qnr_parser._parse_docx` reads docx in document order (paragraphs + tables interleaved)

## Current state (last updated 2026-04-10, session 18)
- Variable Mapping page: table layout per question, draggable option chips, auto-assignment, Save button
- Stage 0: duplicate s0-upload-status ID fixed; page now re-renders immediately after upload (app-state as Input to render_page)
- app.py render_page: app-state changed from State to Input so all pages re-render on state change
- QC fixes: stage0 duplicate ID, var_mapping save-state bugs, vm-state session persistence
- Stages 1–9: inherited from original Streamlit rewrite, not yet re-tested end-to-end

## Open tasks
- Test full pipeline end-to-end with MyRawdata.xlsx + survey.docx
- A7 value=8 (NEE) custom missing value handling (Stage 1)
- Re-zip and upload to Google Drive
