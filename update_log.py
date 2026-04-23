import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from copy import copy

wb = openpyxl.load_workbook('NN_Design_Tracker.xlsx')
ws = wb['Session Log']

next_row = ws.max_row + 1
next_session = 23

decision_text = (
    "Session 2026-04-23 — Fix S10B option truncation + import-options UI. "
    "col_mapper.py: fixed _overlap_score substring collision where short tokens like "
    "'p1' matched inside 'p10', 'p11' etc., causing intersection to drop options 10+. "
    "Changed containment check to word-boundary (set subset) instead of character substring. "
    "Removed early-exit at score>=1.0 in _best_qnr_match so all options are compared; "
    "on equal score, prefer the longer/more specific QNR option. "
    "qnr_parser.py: merge continuation tables split across Word page breaks — when a "
    "second w:tbl element is encountered for the same question, its options are appended "
    "rather than silently dropped (had_table guard relaxed). "
    "Added _cell_is_blue() helper to detect blue-coloured font runs in a table cell. "
    "In _parse_table_for_options, row 1 col 0 is checked: if it is a short question code "
    "in blue text it is stored as borrows_options_from; after full-document dedup pass, "
    "questions with borrows_options_from and no options get the source question's options copied in. "
    "var_mapping.py: pool row now shows an 'Import from...' dcc.Dropdown listing all other "
    "question codes; selecting one runs import_options_from_question callback which appends "
    "source question's options into vm_state['extra_options'][target_code]. "
    "Textarea '+ Add / paste options' panel wired to server: add_options_to_pool callback "
    "reads one-per-line text and saves to extra_options. "
    "Both mechanisms deduplicate and persist across page navigation. "
    "DEFAULT_VM_STATE updated to include extra_options: {}. "
    "Pool construction in _render_var_table includes extra_options deduped with q['options']."
)

open_questions = (
    "A7 value=8 (NEE) custom missing value handling still pending (Stage 1). "
    "Test full pipeline end-to-end with Raw data 2.xlsx + Survey 2.docx. "
    "Re-zip and upload to Google Drive. "
    "Stages 1-9 inherited from Streamlit rewrite, not yet re-tested end-to-end. "
    "borrows_options_from auto-detection from blue PN notes not yet verified against "
    "real survey docx — test with Survey 2.docx to confirm blue-text detection works."
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
ws.cell(row=next_row, column=2).value = '2026-04-23 S23'
ws.cell(row=next_row, column=3).value = 'Fix S10B option truncation + import-options UI — overlap score fix, page-break table merge, borrowed-options auto-detect, Import from / Add to pool callbacks'
ws.cell(row=next_row, column=4).value = decision_text
ws.cell(row=next_row, column=5).value = open_questions

ws.row_dimensions[next_row].height = 200

wb.save('NN_Design_Tracker.xlsx')
print(f"Saved. Session {next_session} added at row {next_row}.")
