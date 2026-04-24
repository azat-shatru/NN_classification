# NN Classification Pipeline — Change Log

## [Session 22] 2026-04-23

### Added
- `col_mapper.py`: Strategy 3 cross-validation in `merge_qnr_with_metadata()` — warns when QNR option count < 70% of what dataset column count implies
- `var_mapping.py`: Red `⚠ incomplete parse` dbc.Badge on accordion card headers for questions with `_mapping_warnings`
- `var_mapping.py`: Inline red alert box inside accordion body showing full warning text

### Changed
- `update_log.py`: Updated for session 22 entry

### Files changed
- `dash_app/utils/col_mapper.py`
- `dash_app/pages/var_mapping.py`
- `update_log.py`
- `CLAUDE.md`

---

## [Session 21] 2026-04-20

### Added / Rewritten
- `col_mapper.py`: `merge_qnr_with_metadata()` — intersection/QNR-only option rule (≥40% word overlap → intersection), grid expansion, zero-unassigned safety pass
- `col_mapper.py`: `parse_excel_metadata()` now reads col B (var_types) in addition to col E (var_labels); case-insensitive sheet lookup
- `var_mapping.py`: `refresh_mapping` calls `merge_qnr_with_metadata` when both QNR + Excel metadata are loaded
- `var_mapping.py`: Type dropdown prefers col-B type over heuristic inference
- `qnr_parser.py`: `_parse_docx` — proper vMerge detection, table-path/no-table-path split, grid-prefix expansion

### Fixed
- `stage0_audit.py`: Duplicate component ID bug fixed
- Pages re-render correctly on `app-state` change

### Files changed
- `dash_app/utils/col_mapper.py`
- `dash_app/utils/qnr_parser.py`
- `dash_app/pages/var_mapping.py`
- `dash_app/pages/stage0_audit.py`
- `update_log.py`
- `CLAUDE.md`

---

## [Sessions 1–20] Pre-2026-04-20

Full 10-stage pipeline built (Stages 0–9). See `NN_Design_Tracker.xlsx` → "Session Log" sheet for detailed per-session notes.

### Stages implemented
- `var_mapping.py` — Pre-stage: upload + interactive tile mapping
- `stage0_audit.py` — Data audit, column editor, drop/type assignment
- `stage1_missing.py` — Missing value imputation
- `stage2_outliers.py` — Outlier detection/handling
- `stage2b_viz.py` — Visualisation dashboard
- `stage3_encoding.py` — Categorical encoding
- `stage4_scaling.py` — Feature scaling
- `stage5_correlation.py` — Correlation analysis
- `stage6_rf.py` — Random Forest feature importance
- `stage7_factor.py` — Factor analysis
- `stage8_combtest.py` — Combined significance testing
- `stage9_nn.py` — PyTorch NN training + evaluation
