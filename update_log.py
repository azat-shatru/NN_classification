import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from copy import copy

wb = openpyxl.load_workbook('NN_Design_Tracker.xlsx')
ws = wb['Session Log']

next_row = ws.max_row + 1
next_session = 15

decision_text = (
    "Session 2026-04-10 — Variable Mapping page full redesign: "
    "(1) Replaced flat AgGrid table with accordion tile view — each QNR question is a collapsible card; "
    "each dataset column is a tile showing column name, type dropdown, 'Move to question' dropdown, and delete (X) button. "
    "(2) Hover tooltip on column name shows first 10 non-null values, unique/missing counts, and QNR answer options. "
    "(3) 'Possibly related' columns detected automatically — columns whose prefix starts with a question code but "
    "with an alphabetic suffix (e.g. A6B related to A6) shown in amber tiles inside that question's card. "
    "(4) Reassign/move: changing the 'Move to question' dropdown reassigns a column to a different question group. "
    "(5) Delete: X button removes a column from the mapping view; 'Reset all' restores deleted/reassigned columns. "
    "(6) Summary row shows Active Columns, Questions Matched, Possibly Related, No QNR Match, QNR Only, Deleted counts. "
    "(7) Sidebar collapse toggle added — hamburger button always visible; clicking collapses sidebar to 48px icon strip "
    "and shifts content area for full-width workspace. State persists in session storage. "
    "(8) Fixed html.A browse links to html.Span in upload areas of var_mapping.py (same fix as stage0)."
)

open_questions = (
    "Test accordion tile view with actual MyRawdata.xlsx + survey.docx. "
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
ws.cell(row=next_row, column=3).value = 'Variable Mapping redesign — accordion tiles, hover preview, delete/move, sidebar collapse'
ws.cell(row=next_row, column=4).value = decision_text
ws.cell(row=next_row, column=5).value = open_questions

ws.row_dimensions[next_row].height = 200

wb.save('NN_Design_Tracker.xlsx')
print(f"Saved. Session {next_session} added at row {next_row}.")
