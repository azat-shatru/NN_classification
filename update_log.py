import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from copy import copy

wb = openpyxl.load_workbook('NN_Design_Tracker.xlsx')
ws = wb['Session Log']

next_row = ws.max_row + 1
next_session = 20

decision_text = (
    "Session 2026-04-17 — Excel metadata auto-mapping for Variable Mapping page. "
    "Added two new functions to col_mapper.py: "
    "(1) parse_excel_metadata(source): reads 'Variable Label Information' and "
    "'Value Label Information' sheets from a data Excel file (accepts file path or BytesIO). "
    "Returns var_labels dict {col_name: label_text} and value_labels dict "
    "{col_name: {numeric_value: label_str}}. Returns has_metadata=False if sheets absent. "
    "(2) build_questions_from_metadata(var_labels, value_labels, col_groups): builds question "
    "dicts in the same format as qnr_parser output. Single-column groups: options come from "
    "value_labels sorted by numeric key. Multi-column groups (multi/grid/multi_col): options "
    "come from stripped var_labels of each column (one chip per column, 'CODE - ' prefix removed). "
    "Type inferred from number of options and value key structure. "
    "Also added _CODE_STRIP_RE regex and _vl_sort_key helper. "
    "var_mapping.py upload_data callback updated: after loading the DataFrame, if file is xlsx/xls, "
    "calls parse_excel_metadata on the raw decoded bytes (BytesIO). If has_metadata=True, "
    "calls build_questions_from_metadata and stores result as qnr_questions in server_store, "
    "sets mapping_done=True immediately (no separate QNR upload needed). "
    "Status message appended with auto-mapped question count. "
    "Raw data 2.xlsx pulled from git remote: has 703 var_labels, 375 value_labels, "
    "builds 76 questions from 'Data with Values' sheet column groups."
)

open_questions = (
    "Test full pipeline end-to-end with Raw data 2.xlsx + Survey 2.docx. "
    "A7 value=8 (NEE) custom missing handling still pending. "
    "Type inference for single-col questions may misclassify categorical as scale_N "
    "(e.g. S1 with 7 options → scale_7); user can override on mapping page. "
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
ws.cell(row=next_row, column=2).value = '2026-04-17 S20'
ws.cell(row=next_row, column=3).value = 'Excel metadata auto-mapping: parse_excel_metadata + build_questions_from_metadata in col_mapper.py; upload_data auto-detects Variable/Value Label sheets'
ws.cell(row=next_row, column=4).value = decision_text
ws.cell(row=next_row, column=5).value = open_questions

ws.row_dimensions[next_row].height = 200

wb.save('NN_Design_Tracker.xlsx')
print(f"Saved. Session {next_session} added at row {next_row}.")
