import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from copy import copy

wb = openpyxl.load_workbook('NN_Design_Tracker.xlsx')
ws = wb['Session Log']

next_row = ws.max_row + 1
next_session = 13

decision_text = (
    "Dash app startup verified — all dependencies installed (kaleido, diskcache, choreographer newly added). "
    "Bug fix in Stage 0 (Data Audit) column editor: previously, columns marked as Drop and applied via 'Apply changes' "
    "button remained visible in the AgGrid column editor with Drop=True, requiring manual visual inspection to confirm removal. "
    "Fix: (1) apply_changes callback now includes a third Output — s0-col-grid rowData — returning only kept rows with Drop reset "
    "to False, so dropped columns disappear from the grid immediately on button press. "
    "(2) flag_sweep callback updated with allow_duplicate=True on its s0-col-grid Output to avoid Dash duplicate output conflict. "
    "(3) layout() function updated to filter out previously-dropped columns (from _col_editor_prior state) when re-rendering "
    "Stage 0, so navigating away and back does not cause dropped columns to reappear in the editor."
)

open_questions = (
    "Test full Dash pipeline end-to-end with MyRawdata.xlsx + survey.docx. "
    "A7 value=8 (NEE) custom missing handling still pending. "
    "Re-zip and upload to Google Drive. "
    "Decide whether to keep Streamlit app as fallback or remove it."
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
ws.cell(row=next_row, column=2).value = '2026-04-05'
ws.cell(row=next_row, column=3).value = 'Stage 0 column editor — drop columns reflect immediately on Apply changes'
ws.cell(row=next_row, column=4).value = decision_text
ws.cell(row=next_row, column=5).value = open_questions

ws.row_dimensions[next_row].height = 200

wb.save('NN_Design_Tracker.xlsx')
print(f"Saved. Session {next_session} added at row {next_row}.")
