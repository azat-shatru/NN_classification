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


# ── text cleaning ──────────────────────────────────────────────────────────────

def _strip_pn(text: str) -> tuple[str, str]:
    """Remove [PN:...] blocks; return (clean_text, raw_pn_text)."""
    pns = PN_RE.findall(text)
    clean = PN_RE.sub("", text).strip()
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean, " | ".join(pns)


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


# ── docx parser ───────────────────────────────────────────────────────────────

def _parse_docx(path: Path) -> list[str]:
    """
    Parse docx in true document order — paragraphs and tables interleaved.
    This preserves the question → options relationship when options live in tables.

    Table handling:
      - Single-cell rows  → one option per line  ("Strongly disagree")
      - Two-cell rows     → value|label pair      ("1" + "Strongly disagree" → "1. Strongly disagree")
      - Multi-cell rows   → join with " | "
    Merged cells are deduplicated via element identity.
    """
    import docx
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph as _Para
    from docx.table import Table as _Table

    doc = docx.Document(str(path))
    lines = []

    for child in doc.element.body:
        tag = child.tag

        # ── paragraph ─────────────────────────────────────────────────────────
        if tag == qn("w:p"):
            t = _Para(child, doc).text.strip()
            if t:
                lines.append(t)

        # ── table ─────────────────────────────────────────────────────────────
        elif tag == qn("w:tbl"):
            tbl = _Table(child, doc)
            seen_tc = set()          # deduplicate merged cells by element identity

            for row in tbl.rows:
                cell_texts = []
                for cell in row.cells:
                    tc = cell._tc
                    if id(tc) in seen_tc:
                        continue
                    seen_tc.add(id(tc))
                    ct = cell.text.strip()
                    if ct:
                        cell_texts.append(ct)

                if not cell_texts:
                    continue

                if len(cell_texts) == 1:
                    # Single column → each row is one option
                    lines.append(cell_texts[0])

                elif len(cell_texts) == 2:
                    # Two columns — common pattern: value | label
                    val, label = cell_texts
                    if re.match(r"^\d+\.?$", val):
                        # "1" + "Strongly disagree" → "1. Strongly disagree"
                        sep = "" if val.endswith(".") else "."
                        lines.append(f"{val}{sep} {label}")
                    else:
                        lines.append(f"{val} | {label}")

                else:
                    # Multi-column: join non-empty cells
                    lines.append(" | ".join(cell_texts))

    return lines


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

    if ext == ".docx":
        lines = _parse_docx(path)
    elif ext == ".pdf":
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
                # Heuristic: short lines after a question are likely options
                if len(clean) < 120 and not clean.startswith("Please") \
                        and not clean.startswith("Your answers") \
                        and not SECTION_RE.match(clean):
                    option_buf.append(clean)
                else:
                    # Append to question text
                    current_q["question"] += " " + clean
                if pn:
                    current_q["pn_notes"] = (current_q.get("pn_notes", "") + " " + pn).strip()

    _flush(current_q, option_buf)

    # Deduplicate by code (keep last occurrence which has most context)
    seen = {}
    for q in questions:
        seen[q["code"]] = q
    return list(seen.values())
