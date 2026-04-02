"""
Auto-maps dataset columns to questionnaire question codes.
Handles:
  - Single columns:   S2, S3, S4  → straight 1:1 mapping
  - Grid columns:     A6_r1_c1, A6_r2_c1 → grouped under A6
  - Multi-select:     S8_1, S8_2, S8_3   → grouped under S8
  - Other suffixes:   _other, _code, _text → attached to parent
"""

import re
import pandas as pd
from collections import defaultdict

# Matches question prefix: S1, A6, B1a, D7a etc.
PREFIX_RE = re.compile(r'^([A-Za-z]\d+[a-z]?)', re.IGNORECASE)

_CHART_OPTIONS = {
    "categorical": ["Bar chart", "Pie chart", "Donut chart"],
    "ordinal":     ["Bar chart", "Horizontal bar", "Stacked bar", "Line chart"],
    "numeric":     ["Histogram", "Box plot", "Violin plot", "Line chart"],
    "scale_7":     ["Bar chart", "Horizontal bar", "Stacked bar", "Mean line"],
    "scale_5":     ["Bar chart", "Horizontal bar", "Stacked bar", "Mean line"],
    "multi":       ["Bar chart", "Horizontal bar"],
    "grid":        ["Grouped bar", "Heatmap"],
    "open":        ["Word count bar"],
    "single":      ["Bar chart", "Pie chart"],
}


def group_columns(df: pd.DataFrame) -> dict:
    """
    Returns dict:  { 'A6': {'cols': [...], 'group_type': 'grid'|'multi'|'single'},  ... }
    Drops all-null columns automatically.
    """
    # Drop all-null columns
    valid_cols = [c for c in df.columns if df[c].notna().any()]

    groups = defaultdict(list)
    ungrouped = []

    for col in valid_cols:
        m = PREFIX_RE.match(col)
        if m:
            groups[m.group(1).upper()].append(col)
        else:
            ungrouped.append(col)

    result = {}
    for prefix, cols in groups.items():
        # Determine group type
        if len(cols) == 1:
            g_type = "single"
        elif any(re.search(r'_r\d+_c\d+', c) for c in cols):
            g_type = "grid"
        elif all(re.search(r'_\d+$', c) for c in cols):
            g_type = "multi"
        else:
            g_type = "multi_col"

        result[prefix] = {
            "cols":       cols,
            "group_type": g_type,
            "primary":    cols[0],   # main col for simple charts
        }

    # Add ungrouped columns as singles
    for col in ungrouped:
        result[col] = {"cols": [col], "group_type": "single", "primary": col}

    return result


def suggest_var_type(df: pd.DataFrame, cols: list, group_type: str, q_type: str = "") -> str:
    """Suggest variable type from data characteristics."""
    if q_type in ("scale_7", "scale_5"):
        return q_type
    if q_type == "numeric":
        return "numeric"
    if q_type == "open":
        return "open"
    if group_type == "grid":
        return "grid"
    if group_type == "multi":
        return "multi"

    primary = cols[0]
    series = df[primary].dropna()
    if series.empty:
        return "categorical"

    n_unique = series.nunique()
    if pd.api.types.is_float_dtype(series) and n_unique > 10:
        return "numeric"
    if n_unique <= 2:
        return "categorical"
    if n_unique <= 7:
        # Check if values are 1–7 integers (likely scale)
        vals = set(pd.to_numeric(series, errors="coerce").dropna().astype(int).unique())
        if vals <= {1, 2, 3, 4, 5, 6, 7}:
            return "scale_7"
        return "ordinal"
    if n_unique <= 15:
        return "ordinal"
    return "numeric"


def build_codebook(df: pd.DataFrame, parsed_questions: list) -> list:
    """
    Merge parsed questionnaire questions with dataset column groups.
    Returns list of codebook dicts ready for Stage 2.5.
    """
    Q_TYPE_TO_VAR = {
        "scale_7": "scale_7", "scale_5": "scale_5",
        "numeric": "numeric", "open": "open",
        "multi": "multi", "grid": "grid", "single": "categorical",
    }

    col_groups  = group_columns(df)
    q_map       = {q["code"].upper(): q for q in parsed_questions}

    codebook = []
    used_prefixes = set()

    # First pass: questions from QNR that have matching dataset columns
    for q in parsed_questions:
        code = q["code"].upper()
        if code not in col_groups:
            continue
        used_prefixes.add(code)
        grp = col_groups[code]

        var_type = suggest_var_type(
            df, grp["cols"], grp["group_type"],
            q_type=Q_TYPE_TO_VAR.get(q["q_type"], "")
        )

        chart_opts = _CHART_OPTIONS.get(var_type, ["Bar chart"])

        codebook.append({
            "code":         code,
            "question":     q["question"],
            "var_type":     var_type,
            "chart_type":   chart_opts[0],
            "group_type":   grp["group_type"],
            "dataset_col":  grp["primary"],
            "all_cols":     grp["cols"],
            "value_labels": {},
            "scale_low":    q.get("scale_low", ""),
            "scale_high":   q.get("scale_high", ""),
            "scale_points": q.get("scale_points"),
            "pn_notes":     q.get("pn_notes", ""),
            "include":      True,
        })

    # Second pass: dataset column groups with no matching QNR question
    for prefix, grp in col_groups.items():
        if prefix in used_prefixes:
            continue
        # Skip system/metadata cols
        if prefix.lower().startswith(("sys_", "vendor", "account", "panel",
                                       "soft", "hcp", "captcha")):
            continue

        var_type = suggest_var_type(df, grp["cols"], grp["group_type"])
        chart_opts = _CHART_OPTIONS.get(var_type, ["Bar chart"])

        codebook.append({
            "code":         prefix,
            "question":     prefix,   # user can rename
            "var_type":     var_type,
            "chart_type":   chart_opts[0],
            "group_type":   grp["group_type"],
            "dataset_col":  grp["primary"],
            "all_cols":     grp["cols"],
            "value_labels": {},
            "scale_low":    "",
            "scale_high":   "",
            "scale_points": None,
            "pn_notes":     "",
            "include":      False,   # off by default if not in QNR
        })

    return codebook
