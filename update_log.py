import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from copy import copy

wb = openpyxl.load_workbook('NN_Design_Tracker.xlsx')
ws = wb['Session Log']

next_row = ws.max_row + 1
next_session = 22

decision_text = (
    "Session 2026-04-23 — QNR parse accuracy: Strategy 3 cross-validation + "
    "incomplete-parse warning in UI. "
    "col_mapper.py: added Step 4b cross-validation block in merge_qnr_with_metadata(). "
    "After final_opts is settled and grid expansion has been attempted, compares "
    "len(final_opts) against len(dataset_cols) / n_col_headers. If QNR options are "
    "less than 70% of what the column count implies (and expected > n_opts), appends "
    "a warning to _mapping_warnings AND emits log.warning() so it surfaces in the "
    "terminal without requiring DEBUG logging. Warning text format: "
    "'{CODE}: QNR has {n} option(s) but {m} dataset column(s) suggest ~{expected} "
    "— QNR parsing may be incomplete (check table structure in the source document)'. "
    "var_mapping.py: accordion card header now shows a red '⚠ incomplete parse' "
    "dbc.Badge for any question whose _mapping_warnings contains the parse-incomplete "
    "signal; badge title tooltip shows the full message. Inside the accordion body, "
    "a red inline alert box (background #fef2f2, border #fca5a5) renders the full "
    "warning text so the user sees the mismatch count without hovering. "
    "Survey 2.docx updated by user (re-uploaded). "
    "Raw data 2.xlsx updated (minor binary change)."
)

open_questions = (
    "S10B root cause still unknown — strategy 3 will now surface the exact mismatch "
    "count when Survey 2.docx + Raw data 2.xlsx are loaded in the app. "
    "Investigate the actual QNR docx structure for S10B once warning fires: "
    "check whether it is a page-break split table (Strategy 1) or vMerge skip "
    "on rows 10-20 (Strategy 2). "
    "A7 value=8 (NEE) custom missing value handling still pending (Stage 1). "
    "Test full pipeline end-to-end with Raw data 2.xlsx + Survey 2.docx. "
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
ws.cell(row=next_row, column=2).value = '2026-04-23 S22'
ws.cell(row=next_row, column=3).value = 'QNR parse accuracy: Strategy 3 cross-validation — col_mapper warns when option count < dataset column count; red badge + alert in Var Mapping UI'
ws.cell(row=next_row, column=4).value = decision_text
ws.cell(row=next_row, column=5).value = open_questions

ws.row_dimensions[next_row].height = 200

wb.save('NN_Design_Tracker.xlsx')
print(f"Saved. Session {next_session} added at row {next_row}.")
