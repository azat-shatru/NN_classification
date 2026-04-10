import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from copy import copy

wb = openpyxl.load_workbook('NN_Design_Tracker.xlsx')
ws = wb['Session Log']

next_row = ws.max_row + 1
next_session = 16

decision_text = (
    "Session 2026-04-10 (continued) — Variable Mapping further improvements: "
    "(1) Answer options moved from card-level panel into each individual variable tile — "
    "each tile now shows its question's options as indigo pill chips (max 6 visible + '+N more'). "
    "Scale questions show '1 = low → N = high' anchor text instead. "
    "(2) qnr_parser.py _parse_docx rewritten to read docx in true document order (paragraphs and tables interleaved). "
    "Previously all paragraphs were read first then all tables, breaking the question→options link when options "
    "live in Word tables. Fix uses doc.element.body iteration with qn('w:p') / qn('w:tbl') tags. "
    "Handles three table layouts: single-cell rows (one option per row), two-cell value|label rows "
    "(formats as '1. Label'), and multi-cell rows (joined with ' | '). Merged cells deduplicated by element identity."
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
ws.cell(row=next_row, column=2).value = '2026-04-10'
ws.cell(row=next_row, column=3).value = 'Var Mapping: inline options per tile + qnr_parser docx table-order fix'
ws.cell(row=next_row, column=4).value = decision_text
ws.cell(row=next_row, column=5).value = open_questions

ws.row_dimensions[next_row].height = 200

wb.save('NN_Design_Tracker.xlsx')
print(f"Saved. Session {next_session} added at row {next_row}.")
