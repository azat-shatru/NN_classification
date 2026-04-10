import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from copy import copy

wb = openpyxl.load_workbook('NN_Design_Tracker.xlsx')
ws = wb['Session Log']

next_row = ws.max_row + 1
next_session = 18

decision_text = (
    "Session 2026-04-10 — Var Mapping redesign + QC fixes. "
    "(1) Variable Mapping page fully redesigned: replaced per-variable accordion tiles with a "
    "4-column table per question (Variable | Options | Type | Question). Options rendered as "
    "draggable HTML chips (opt-chip class); drag-and-drop handled by assets/dragdrop.js using "
    "document-level event delegation so listeners survive Dash re-renders. Drop zones (opt-dropzone) "
    "highlight on hover. Auto-assignment logic: single col → all opts; multi q_type + numeric/letter "
    "suffix → indexed opt; grid → all opts per col; scale/numeric → no chips. Unassigned options "
    "pool row shown at table bottom. Save button triggers clientside_callback that reads DOM chip "
    "positions into vm-drag-assignments store, then handle_save server callback commits all changes. "
    "vm-state storage changed to session so assignments survive page navigation. handle_save guards "
    "against empty drag_assigns dict overwriting saved option_assignments. "
    "(2) Stage 0 QC fix: removed duplicate id=s0-upload-status element (was at lines 103 and 159). "
    "(3) app.py QC fix: render_page app-state changed from State to Input so pages re-render "
    "immediately after upload without requiring navigation. "
    "New file: dash_app/assets/dragdrop.js. CSS additions: opt-chip, opt-dropzone, opt-chip-pool styles."
)

open_questions = (
    "Test full pipeline end-to-end with MyRawdata.xlsx + survey.docx. "
    "A7 value=8 (NEE) custom missing handling still pending. "
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
ws.cell(row=next_row, column=2).value = '2026-04-10'
ws.cell(row=next_row, column=3).value = 'Var Mapping: table+drag-drop redesign; Stage 0 + app.py QC fixes'
ws.cell(row=next_row, column=4).value = decision_text
ws.cell(row=next_row, column=5).value = open_questions

ws.row_dimensions[next_row].height = 200

wb.save('NN_Design_Tracker.xlsx')
print(f"Saved. Session {next_session} added at row {next_row}.")
