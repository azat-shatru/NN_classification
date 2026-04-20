import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from copy import copy

wb = openpyxl.load_workbook('NN_Design_Tracker.xlsx')
ws = wb['Session Log']

next_row = ws.max_row + 1
next_session = 21

decision_text = (
    "Session 2026-04-20 — Accurate variable-to-option assignment using Excel metadata + QNR. "
    "col_mapper.py: parse_excel_metadata now reads col B (variable type) into var_types dict and "
    "uses case-insensitive sheet name lookup. Added _col_b_to_var_type() helper mapping raw col-B "
    "strings to canonical var-type keys. Added merge_qnr_with_metadata(questions, meta, col_groups) "
    "as the central enrichment function. Logic: (1) for each question, collect col-E labels from "
    "'Variable Label Information' sheet for all matched columns; (2) test for overlap between "
    "QNR options and col-E labels using substring containment and word-overlap ratio >= 0.4; "
    "(3) if overlap found → use INTERSECTION only, preserving QNR text as canonical (no duplicate "
    "option versions); if no overlap → use QNR options only; (4) grid expansion fires when "
    "n_matched_vars == n_options * n_col_headers, in three detection modes: QNR-provided "
    "table_col_headers, headers inferred from variable name suffixes, legacy table_n_cols > 2; "
    "expanded options formatted as '{col_header} {option}' assigned sequentially; (5) safety pass "
    "ensures every column gets an assignment — suffix positional fallback (_1 → opts[0]) then "
    "overassign all options if still unresolved; (6) annotates each enriched question with "
    "_mapping_source and _mapping_warnings for auditability; (7) calls log_assignments() at DEBUG "
    "level for per-column assignment tracing. "
    "var_mapping.py: _default_option_assignments reordered so col_assignments branch runs before "
    "scale/numeric short-circuit (metadata assignments survive for all q_types); grid expansion "
    "block generalised to fire on col_headers length match, not just table_n_cols > 2; final "
    "safety pass added after all loops to guarantee no column is left blank. refresh_mapping "
    "callback now calls merge_qnr_with_metadata when both QNR and Excel metadata are loaded. "
    "Type dropdown now prefers meta_var_type (col B) over heuristic suggest_var_type. "
    "Test result: 351 columns enriched, 0 unassigned across all matched questions."
)

open_questions = (
    "S10B (and similar suffixed variants like S10b_N_1) not parsed from QNR docx — "
    "parser stops at S10A; these fall back to Excel metadata only. QNR parser fix pending. "
    "A7 value=8 (NEE) custom missing handling still pending (Stage 1). "
    "Test full pipeline end-to-end with Raw data 2.xlsx + survey.docx. "
    "Re-zip and upload to Google Drive. "
    "Stages 1-9 inherited from Streamlit rewrite, not yet re-tested end-to-end."
)


# Copy style from previous data row
def copy_row_style(src_row_num, dst_row_num):
    for col in range(1, ws.max_column + 1):
        src_cell = ws.cell(row=src_row_num, column=col)
        dst_cell = ws.cell(row=dst_row_num, column=col)
        if src_cell.has_style:
            dst_cell.font = copy(src_cell.font)
            dst_cell.fill = copy(src_cell.fill)
            dst_cell.border = copy(src_cell.border)
            dst_cell.alignment = copy(src_cell.alignment)
            dst_cell.number_format = src_cell.number_format

copy_row_style(ws.max_row, next_row)

ws.cell(row=next_row, column=1).value = next_session
ws.cell(row=next_row, column=2).value = '2026-04-20 S21'
ws.cell(row=next_row, column=3).value = 'Var Mapping: accurate option assignment via QNR+Excel merge — intersection rule, grid expansion, zero-unassigned safety pass, col-B type, debug logging'
ws.cell(row=next_row, column=4).value = decision_text
ws.cell(row=next_row, column=5).value = open_questions

ws.row_dimensions[next_row].height = 200

wb.save('NN_Design_Tracker.xlsx')
print(f"Saved. Session {next_session} added at row {next_row}.")
