import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from copy import copy

wb = openpyxl.load_workbook('NN_Design_Tracker.xlsx')
ws = wb['Session Log']

next_row = ws.max_row + 1
next_session = 19

decision_text = (
    "Session 2026-04-17 — qnr_parser docx rewrite + var_mapping grid-prefix expansion + B3 flush bug fix. "
    "(1) _parse_docx fully rewritten to return list[dict] directly instead of list[str]. "
    "New _parse_table_for_options helper: skips row 0 always; reads ONLY first column; "
    "skips vertical-merge continuation rows (w:vMerge without val=restart); "
    "skips horizontally-merged first-cell rows (w:gridSpan > 1); "
    "skips single-cell disclaimer rows; extracts table_col_headers (row 0 cols 1+) and table_n_cols. "
    "Two-pass approach: pass 1 collects (para, text) and (table, obj) in document order; "
    "pass 2 groups into questions. Table path: buffered paragraphs before table → question text, "
    "table → options. No-table path: all paragraphs except last → question text, last → single option "
    "(unless it starts with Please/Your answers or is all-caps section header). "
    "Validation warning printed if extracted options < 90% of unmerged row count. "
    "parse_questionnaire short-circuits for .docx, returning _parse_docx result directly. "
    "PDF/xlsx/txt paths unchanged. "
    "(2) _default_option_assignments in var_mapping.py: added grid-prefix expansion. "
    "For questions with table_n_cols > 2 and table_col_headers present, if n_matched_vars "
    "approximately equals n_opts × (n_table_cols - 1) within ±10%, expands options as "
    "'col_header opt' for each option × each column header, assigned 1:1 to matched+extra cols. "
    "(3) Bug fix: _flush_question called with [] instead of text_buf when a post-table paragraph "
    "appeared before the next question code — pre-table paragraphs were silently discarded. "
    "Fixed by passing text_buf in the had_table branch (qnr_parser.py line ~336)."
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
ws.cell(row=next_row, column=2).value = '2026-04-17'
ws.cell(row=next_row, column=3).value = 'qnr_parser docx rewrite: proper merge detection, table-path/no-table-path, grid-prefix expansion, B3 flush bug fix'
ws.cell(row=next_row, column=4).value = decision_text
ws.cell(row=next_row, column=5).value = open_questions

ws.row_dimensions[next_row].height = 200

wb.save('NN_Design_Tracker.xlsx')
print(f"Saved. Session {next_session} added at row {next_row}.")
