import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from copy import copy

wb = openpyxl.load_workbook('NN_Design_Tracker.xlsx')
ws = wb['Session Log']

next_row = ws.max_row + 1
next_session = 17

decision_text = (
    "Session 2026-04-10 — Var Mapping: interactive option assignment per variable tile. "
    "Replaced static read-only answer-option chips on each variable tile with an interactive "
    "multi-select dropdown. Each tile now shows an 'Assigned options' dropdown populated with "
    "the parent question's answer options from the QNR; user can select one or more options to "
    "indicate which answer options that variable represents. Selections stored in "
    "vm_state['option_assignments'] (dict: col -> list of selected option strings) and persist "
    "within the session. Reset all clears assignments along with other vm_state fields. "
    "Callback: handle_option_assign uses pattern-matching Input({'type': 'vm-opt-sel', 'index': ALL}). "
    "DEFAULT_VM_STATE updated to include 'option_assignments': {}. "
    "Committed as 4f8dea8 and pushed to origin/dash-app."
)

open_questions = (
    "Test full pipeline end-to-end with MyRawdata.xlsx + survey.docx. "
    "A7 value=8 (NEE) custom missing handling still pending. "
    "Re-zip and upload to Google Drive. "
    "Continue logging all code changes in update_log.py per session."
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
ws.cell(row=next_row, column=2).value = '2026-04-10'  # session 17
ws.cell(row=next_row, column=3).value = 'Var Mapping: interactive multi-select option assignment per variable tile'
ws.cell(row=next_row, column=4).value = decision_text
ws.cell(row=next_row, column=5).value = open_questions

ws.row_dimensions[next_row].height = 200

wb.save('NN_Design_Tracker.xlsx')
print(f"Saved. Session {next_session} added at row {next_row}.")
