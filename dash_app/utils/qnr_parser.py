"""
Questionnaire parser — extracts question codes, text, type hints and
answer options from Word (.docx), PDF, Excel, or plain-text files.

Returns a list of dicts:
  {
    "code":        "A6",
    "full_code":   "A6",
    "question":    "Please rate your level of knowledge...",
    "q_type":      "scale_7" | "numeric" | "single" | "multi" | "open" | "grid",
    "scale_low":   "Not at all Knowledgeable",   # if scale
    "scale_high":  "Extremely Knowledgeable",    # if scale
    "scale_points": 7,
    "options":     ["Option A", "Option B", ...],
    "pn_notes":    "raw programmer note text",
  }
"""

import re
from pathlib import Path


# ── regex patterns ─────────────────────────────────────────────────────────────

# Matches: S1. / S1a. / A4. / B1b. / C2. / D7a. / E3. / F2.
Q_CODE_RE = re.compile(
    r"^\s*([A-Z]\d{1,2}[a-z]?(?:\d)?)\s*[.\t]\s*(.+)", re.IGNORECASE
)

# Programmer notes [PN: ...]
PN_RE = re.compile(r"\[PN:.*?\]", re.DOTALL | re.IGNORECASE)

# 7-point / n-point scale hints
SCALE_RE  = re.compile(r"(\d+)-point scale", re.IGNORECASE)
ANCHOR_RE = re.compile(
    r'(?:where\s+)?1\s*=\s*["\u201c\u2018]([^"\u201d\u2019]+)["\u201d\u2019]\s+and\s+\d\s*=\s*["\u201c\u2018]([^"\u201d\u2019]+)["\u201d\u2019]',
    re.IGNORECASE
)

# Range hints  [Range 0 - 9999]
RANGE_RE = re.compile(r"range[:\s]+(\d+)\s*[-–]\s*(\d+)", re.IGNORECASE)

# Select hints
MULTI_RE  = re.compile(r"select all that apply|allow multiple", re.IGNORECASE)
SINGLE_RE = re.compile(r"allow only one|one selection", re.IGNORECASE)
OPEN_RE   = re.compile(r"open end|open-end|text box|list all", re.IGNORECASE)
GRID_RE   = re.compile(r"one selection per row|per row|grid", re.IGNORECASE)

# Section headers (no question code — all-caps or title-only lines)
SECTION_RE = re.compile(r"^[A-Z][A-Z\s\-/&.,]+$")

# Table header rows to skip (column labels that are not answer options)
TABLE_HEADER_RE = re.compile(
    r"^(answer\s*code|answer\s*label|code|label|option|answer|response|"
    r"category|item|description|statement|variable|value|score|choice)s?\s*$",
    re.IGNORECASE
)


# ── text cleaning ──────────────────────────────────────────────────────────────

def _strip_pn(text: str) -> tuple[str, str]:
    """Remove [PN:...] blocks; return (clean_text, raw_pn_text)."""
    pns = PN_RE.findall(text)
    clean = PN_RE.sub("", text).strip()
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean, " | ".join(pns)


def _cell_is_blue(cell) -> bool:
    """Return True if any run in the cell has a blue-ish font color."""
    from docx.oxml.ns import qn
    for para in cell.paragraphs:
        for run in para.runs:
            rPr = run._r.find(qn("w:rPr"))
            if rPr is None:
                continue
            color_el = rPr.find(qn("w:color"))
            if color_el is None:
                continue
            val = color_el.get(qn("w:val"), "")
            if not val or val.lower() == "auto":
                continue
            try:
                r = int(val[0:2], 16)
                g = int(val[2:4], 16)
                b = int(val[4:6], 16)
                if b > 100 and b > r and b > g:
                    return True
            except (ValueError, IndexError):
                pass
    return False


def _detect_type(question_text: str, pn_text: str, options: list) -> dict:
    """Infer question type and scale info from text."""
    combined = question_text + " " + pn_text

    result = {
        "q_type":       "single",
        "scale_points": None,
        "scale_low":    "",
        "scale_high":   "",
        "range_min":    None,
        "range_max":    None,
    }

    # Scale
    sm = SCALE_RE.search(combined)
    if sm:
        result["q_type"]       = f"scale_{sm.group(1)}"
        result["scale_points"] = int(sm.group(1))

    am = ANCHOR_RE.search(combined)
    if am:
        result["scale_low"]  = am.group(1).strip()
        result["scale_high"] = am.group(2).strip()

    # Numeric range
    rm = RANGE_RE.search(combined)
    if rm:
        result["q_type"]    = "numeric"
        result["range_min"] = int(rm.group(1))
        result["range_max"] = int(rm.group(2))

    # Open end
    if OPEN_RE.search(combined):
        result["q_type"] = "open"

    # Multi-select
    if MULTI_RE.search(combined):
        result["q_type"] = "multi"

    # Grid
    if GRID_RE.search(combined) and result["q_type"] not in ("scale_7", "scale_5"):
        result["q_type"] = "grid"

    return result


# ── docx table helper ────────────────────────────────────────────────────────

def _parse_table_for_options(tbl) -> dict:
    """
    Extract options from a docx table object.
    Returns {"options": [...], "table_col_headers": [...],
             "table_n_cols": int, "n_unmerged_rows": int}.
    """
    from docx.oxml.ns import qn

    rows = tbl.rows
    if not rows:
        return {"options": [], "table_col_headers": [], "table_n_cols": 0, "n_unmerged_rows": 0}

    # ── Step 0: single-cell table (disclaimer) detection ─────────────────────
    if len(rows) == 1:
        return {"options": [], "table_col_headers": [], "table_n_cols": 1, "n_unmerged_rows": 0}

    # Check if every row is a single merged cell
    all_single = True
    for row in rows:
        unique_tcs = []
        for cell in row.cells:
            if not unique_tcs or cell._tc is not unique_tcs[-1]:
                unique_tcs.append(cell._tc)
        if len(unique_tcs) > 1:
            all_single = False
            break
    if all_single:
        return {"options": [], "table_col_headers": [], "table_n_cols": 1, "n_unmerged_rows": 0}

    # ── Step 1: read column count and headers from row 0 ─────────────────────
    row0 = rows[0]
    unique_tcs_r0 = []
    for cell in row0.cells:
        if not unique_tcs_r0 or cell._tc is not unique_tcs_r0[-1]:
            unique_tcs_r0.append(cell._tc)
    n_cols = len(unique_tcs_r0)

    # Header texts: skip col 0, take col 1+
    header_texts = []
    for tc in unique_tcs_r0:
        # Get text from the tc element
        texts = []
        for p in tc.findall(qn("w:p")):
            t = "".join(node.text or "" for node in p.iter() if node.tag == qn("w:t"))
            texts.append(t)
        header_texts.append(" ".join(texts).strip())
    table_col_headers = header_texts[1:]  # skip col 0

    # ── Step 1b: detect "borrows options from" reference ─────────────────────
    # Row 1, col 0 written in blue = PN note pointing to source question code
    borrows_from = None
    if len(rows) > 1:
        row1_cells = rows[1].cells
        if row1_cells:
            cell_text = row1_cells[0].text.strip()
            if (re.match(r'^[A-Z]\d{1,2}[a-z]?\d?$', cell_text, re.IGNORECASE)
                    and _cell_is_blue(row1_cells[0])):
                borrows_from = cell_text.upper()

    # ── Step 2: iterate rows 1+ for options ──────────────────────────────────
    options = []
    n_unmerged_rows = 0

    for row_idx in range(1, len(rows)):
        row = rows[row_idx]
        cells = row.cells
        if not cells:
            continue

        first_tc = cells[0]._tc

        # Skip: vertical merge continuation on first cell
        vmerge = first_tc.find(qn("w:tcPr"))
        if vmerge is not None:
            vm_el = vmerge.find(qn("w:vMerge"))
            if vm_el is not None and vm_el.get(qn("w:val")) != "restart":
                continue

        # Skip: horizontal merge on first cell (gridSpan > 1)
        if vmerge is not None:
            grid_span = vmerge.find(qn("w:gridSpan"))
            if grid_span is not None:
                span_val = grid_span.get(qn("w:val"))
                if span_val and int(span_val) > 1:
                    continue

        # Skip: single-cell row spanning all columns
        unique_tcs_row = []
        for cell in cells:
            if not unique_tcs_row or cell._tc is not unique_tcs_row[-1]:
                unique_tcs_row.append(cell._tc)
        if len(unique_tcs_row) == 1 and n_cols > 1:
            continue

        # Not skipped → count as unmerged
        n_unmerged_rows += 1

        text = cells[0].text.strip()
        if not text:
            continue
        if TABLE_HEADER_RE.match(text):
            continue

        options.append(text)

    # Validation logging
    if n_unmerged_rows > 0 and len(options) < (n_unmerged_rows - 1) * 0.9 - 1:
        print(f"[qnr_parser] WARNING: only {len(options)} options extracted from "
              f"{n_unmerged_rows} unmerged rows — check table structure")

    return {
        "options": options,
        "table_col_headers": table_col_headers,
        "table_n_cols": n_cols,
        "n_unmerged_rows": n_unmerged_rows,
        "borrows_options_from": borrows_from,
    }


# ── docx parser ───────────────────────────────────────────────────────────────

def _parse_docx(path: Path) -> list[dict]:
    """
    Parse docx in true document order — paragraphs and tables interleaved.
    Returns list[dict] of question dicts directly (no intermediate line list).
    """
    import docx
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph as _Para
    from docx.table import Table as _Table

    doc = docx.Document(str(path))

    # ── Pass 1: collect elements in document order ───────────────────────────
    elements = []
    for child in doc.element.body:
        tag = child.tag
        if tag == qn("w:p"):
            t = _Para(child, doc).text.strip()
            if t:
                elements.append(("para", t))
        elif tag == qn("w:tbl"):
            elements.append(("table", _Table(child, doc)))

    # ── Pass 2: group into questions ─────────────────────────────────────────
    questions = []
    current_q = None
    text_buf: list[str] = []    # paragraphs after the Q_CODE line
    had_table = False           # whether a table was already consumed for current_q

    NON_OPTION_STARTS = ("Please", "Your answers")

    def _flush_question(q, buf, table_info):
        """Finalize and append a question dict."""
        if not q:
            return

        if table_info is not None:
            # Table path: all buffered paragraphs go into question text
            for line in buf:
                q["question"] = (q["question"] + " " + line).strip()
            q["options"] = table_info["options"]
            q["table_col_headers"] = table_info["table_col_headers"]
            q["table_n_cols"] = table_info["table_n_cols"]
            if table_info.get("borrows_options_from"):
                q["borrows_options_from"] = table_info["borrows_options_from"]
        else:
            # No-table path
            if buf:
                last = buf[-1]
                # Append all but last to question text
                for line in buf[:-1]:
                    q["question"] = (q["question"] + " " + line).strip()
                # Last paragraph: treat as option unless it looks like instructions
                if (last.startswith(tuple(NON_OPTION_STARTS))
                        or SECTION_RE.match(last)):
                    q["question"] = (q["question"] + " " + last).strip()
                    q["options"] = []
                else:
                    q["options"] = [last]
            else:
                q["options"] = []

        # Detect type and merge
        info = _detect_type(q["question"], q.get("pn_notes", ""), q["options"])
        q.update(info)
        questions.append(q)

    table_info_for_current = None

    for kind, val in elements:
        if kind == "para":
            clean, pn = _strip_pn(val)
            if not clean and not pn:
                continue

            m = Q_CODE_RE.match(clean) if clean else None
            if m:
                # Flush previous question
                _flush_question(current_q, text_buf,
                                table_info_for_current if had_table else None)

                code = m.group(1).upper()
                text = m.group(2).strip()
                text, pn2 = _strip_pn(text)

                current_q = {
                    "code": code,
                    "full_code": code,
                    "question": text,
                    "pn_notes": (pn + " " + pn2).strip(),
                }
                text_buf = []
                had_table = False
                table_info_for_current = None
            elif current_q:
                # Accumulate PN notes
                if pn:
                    current_q["pn_notes"] = (
                        current_q.get("pn_notes", "") + " " + pn
                    ).strip()
                if clean:
                    if had_table:
                        # After a table, paragraphs go to NEXT question's buffer
                        # — but since no new Q_CODE yet, we start a fresh buffer
                        # for when the next question eventually starts.
                        # Actually, these paragraphs just get buffered and will
                        # be part of the next question once it starts. For now
                        # we still append to text_buf which will be flushed when
                        # the next Q_CODE is seen; they'll go into the no-table
                        # path for the NEXT question. But current_q already has
                        # its table info. So we need to flush current_q first.
                        _flush_question(current_q, text_buf,
                                        table_info_for_current)
                        current_q = None
                        had_table = False
                        table_info_for_current = None
                        text_buf = [clean]
                    else:
                        text_buf.append(clean)

        elif kind == "table":
            if current_q:
                new_info = _parse_table_for_options(val)
                if not had_table:
                    table_info_for_current = new_info
                    had_table = True
                else:
                    # Continuation table split across a page break — append options
                    table_info_for_current["options"].extend(new_info["options"])

    # Flush last question
    _flush_question(current_q, text_buf,
                    table_info_for_current if had_table else None)

    # Deduplicate by code (keep last occurrence)
    seen = {}
    for q in questions:
        seen[q["code"]] = q

    # Resolve borrowed options: questions whose table pointed to another question's code
    q_opt_map = {code: q.get("options", []) for code, q in seen.items()}
    for q in seen.values():
        bf = q.get("borrows_options_from")
        if bf and not q.get("options"):
            src_opts = q_opt_map.get(bf.upper(), [])
            if src_opts:
                q["options"] = list(src_opts)

    return list(seen.values())


def _parse_pdf(path: Path) -> list[str]:
    import pdfplumber
    lines = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split("\n"):
                    t = line.strip()
                    if t:
                        lines.append(t)
    return lines


def _parse_txt(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8", errors="ignore")
            .splitlines() if l.strip()]


def _parse_excel(path: Path) -> list[str]:
    import openpyxl
    wb = openpyxl.load_workbook(str(path), data_only=True)
    lines = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for cell in row:
                if cell:
                    lines.append(str(cell).strip())
    return lines


# ── main parser ───────────────────────────────────────────────────────────────

def parse_questionnaire(path) -> list[dict]:
    """
    Parse a questionnaire file and return a list of question dicts.
    """
    path = Path(path)
    ext  = path.suffix.lower()

    # docx uses its own structured parser that returns list[dict] directly
    if ext == ".docx":
        return _parse_docx(path)

    if ext == ".pdf":
        lines = _parse_pdf(path)
    elif ext in (".xlsx", ".xls"):
        lines = _parse_excel(path)
    else:
        lines = _parse_txt(path)

    questions   = []
    current_q   = None
    option_buf  = []   # buffer for answer options between questions

    def _flush(q, opts):
        if q:
            q["options"] = opts[:]
            info = _detect_type(q["question"], q.get("pn_notes", ""), opts)
            q.update(info)
            questions.append(q)

    for line in lines:
        clean, pn = _strip_pn(line)
        if not clean:
            continue

        m = Q_CODE_RE.match(clean)
        if m:
            # Save previous question
            _flush(current_q, option_buf)
            option_buf = []

            code = m.group(1).upper()
            text = m.group(2).strip()
            # Remove trailing PN from question text
            text, pn2 = _strip_pn(text)

            current_q = {
                "code":      code,
                "full_code": code,
                "question":  text,
                "pn_notes":  (pn + " " + pn2).strip(),
            }
        elif current_q:
            # Could be a scale anchor, option, or continuation of question text
            if pn and not clean:
                current_q["pn_notes"] = (current_q.get("pn_notes", "") + " " + pn).strip()
            elif clean:
                # Table-header-like lines are silently skipped
                if TABLE_HEADER_RE.match(clean):
                    pass
                # Everything else after a question becomes an option —
                # no length limit since options can be complete sentences
                elif not clean.startswith("Please") \
                        and not clean.startswith("Your answers") \
                        and not SECTION_RE.match(clean):
                    option_buf.append(clean)
                else:
                    current_q["question"] += " " + clean
                if pn:
                    current_q["pn_notes"] = (current_q.get("pn_notes", "") + " " + pn).strip()

    _flush(current_q, option_buf)

    # Deduplicate by code (keep last occurrence which has most context)
    seen = {}
    for q in questions:
        seen[q["code"]] = q
    return list(seen.values())
