"""Salary `.xlsx` parser — extract-and-return (ADR-0010 pattern, ADR-0014).

hr-ai parses the workbook and RETURNS structured rows; hr-backend writes
`salary_tables` / `salary_table_rows` / `convenio_job_categories`. hr-ai writes
NO salary DB rows.

Handles the real messiness seen across COEAS Andalucía / Deporte Cantabria /
COEAS Estatal:
- ignore junk/notes sheets (no salary-grid header, or tiny);
- the header row is NOT row 1 (scan for it);
- map cryptic columns via a header-synonym map (per-format maps converge on one
  synonym set);
- ONE workbook → MANY year tables (e.g. `smi 26` + `smi 25`), one per sheet
  (ADR-0014 / plan §9 Q3).

EVERY typed figure comes from a source cell (ADR-0006). Correction-salary-01
removed the previous "canonical 14/12 mapping", which derived
`base_salary_monthly = gross_annual / 14` and asserted `num_payments = 14` for
every table: a convenio that does not pay in 14 was told a monthly figure its
own gazette contradicts (convenio 15, Gestores Información Gipuzkoa: gazette
2.232,75 €, derived 2.392,24 € — 15 pagas, not 14). Now:

- `base_salary_monthly` is set ONLY from a column the source itself labels as a
  monthly base (`SB`, `Salario base`, `Salario base (mes)`, `14 pagas`…), never
  computed from the annual;
- `pagas_count` is a typed field of its own, set ONLY when a header states it
  ("14 pagas", "Bruto/mes 12 pagas"), and is NEVER used to derive anything;
- when the source states no monthly, `base_salary_monthly` stays NULL and the
  answer says the annual instead of inventing a monthly.

ALL original columns (SB, COMP, Comp. SMI, the 14 & 12 figures, totals…) are
kept verbatim in raw_values regardless. Documented in data-model.md §6.
"""

from __future__ import annotations

import io
import re
import unicodedata

import openpyxl

# Typed-column numeric bounds (must match hr-backend's salary_table_rows
# migration precision — decimal(10,2) for the money columns, decimal(8,4) for
# hourly_rate) — found live on staging (2026-09-06): a real spreadsheet
# ("Tablas Intervencion Social Navarra", another "Tablas COEAS Navarra") has
# a column genuinely header-labeled "€/hora" whose value is actually an
# annual figure (13448.62 == 12 × the adjacent year's monthly figure) — a
# data-entry error in the SOURCE spreadsheet, not a column-mapping bug here
# (the header match is correct). Inserting it verbatim overflowed
# hourly_rate's decimal(8,4) column and crashed the whole salary:import run
# for every remaining document (ADR-0014's per-document isolation only
# covers extractSalary() failures, not a DB-constraint violation inside the
# write transaction). Rather than crash OR silently guess a "corrected"
# value, an out-of-range typed value is dropped (kept in raw_values
# verbatim, same as every other column) and reported in `warnings` — visible,
# not guessed, per ADR-0014's whole design.
_FIELD_BOUNDS = {
    "gross_annual": 99_999_999.99,
    "base_salary_monthly": 99_999_999.99,
    "extra_pay": 99_999_999.99,
    "hourly_rate": 9_999.9999,
    "night_plus": 99_999_999.99,
}


def _norm(text) -> str:
    if text is None:
        return ""
    s = str(text).replace("\n", " ")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.strip(" .:·-")


# Header-synonym sets (normalized) for the TYPED columns we extract. These
# converge across the real per-format files (COEAS Andalucía/Estatal, Deporte
# Cantabria/Estatal, Agencias, Limpieza Navarra, …) — plan §9 Q2.
_GROSS = {"total", "total anual", "bruto anual", "bruto ano", "bruto año", "importe anual", "salario anual"}
_HOURLY = {
    "€/hora", "e/hora", "euro/hora", "euros/hora", "hora", "precio hora", "precio/hora",
    "coste hora", "/hora", "bruto/hora", "bruto hora", "salario hora",
}
_EXTRA = {"pagas extra", "paga extra", "pagas extras", "paga extras"}
# Headers whose cell IS a stated monthly BASE salary (Correction-salary-01).
# Strictly base-monthly: "bruto mes" / "bruto/mes 14 pagas" are a GROSS monthly
# (base + prorated extras + pluses), a different concept from the
# `base_salary_monthly` column, so they stay in raw_values only — though the
# pagas count they state is still read (see `_stated_pagas`).
_MONTHLY = {
    "sb", "salario base", "salario base mensual", "sueldo base", "sueldo mensual",
    "salario mensual", "base mensual", "salario mes", "salario/mes", "base mes",
}
# A header that states how many payments the table is expressed over: "14
# pagas", "Bruto/mes 12 pagas", "(14 pagas)". Requires the count BEFORE the
# word, so a column labelled "PAGA 16" (Limpieza Navarra — the value of one of
# 16 payments, not a payment count) is deliberately NOT matched.
_PAGAS_RE = re.compile(r"\b(\d{1,2})\s*pagas\b")
# A column that is itself a monthly amount stated at a given pagas count, e.g.
# COEAS Navarra's side-by-side "14 pagas" / "12 pagas" columns.
_MONTHLY_AT_PAGAS_RE = re.compile(r"^(\d{1,2})\s*pagas$")
_NIGHT = {"plus nocturno", "nocturnidad", "plus noche", "nocturno", "plus nocturnidad",
          "plus hora nocturna", "plus hora noctur", "hora nocturna"}

# Raw money-column markers (kept verbatim in raw_values, used to anchor where the
# numeric grid starts — everything LEFT of the first money header is a label).
# ("sb" and "salario base" moved to _MONTHLY — they are now TYPED, not raw-only.
# The union below is unchanged, so header detection and the label/grid boundary
# behave exactly as before.)
_RAW_MONEY = {
    "sb anual", "comp", "comp.", "comp smi", "comp. smi", "comp smi / ano",
    "comp smi / mes", "14", "12", "bruto mes", "bruto/mes 14 pagas", "bruto/mes 12 pagas",
    "dedica", "pc", "paga 16", "p.p.paga extra", "plus transporte",
    "plus tpte/dia", "plus tpte/día", "5% mejora sedena", "1,2,3,5 quinquenio",
    "4 quinquenio", "quinquenio", "antiguedad",
}

# A header cell that anchors the start of the numeric grid (typed OR raw money).
_MONEY_HEADERS = _GROSS | _HOURLY | _EXTRA | _NIGHT | _MONTHLY | _RAW_MONEY

# The label column's own header. It carries no figure, but it is what makes a
# TWO-column grid ("Categoría | Salario anual" — the year-column gazette tables
# converted by `salary:pdf-to-xlsx --year-columns`) recognizable as a salary
# grid at all: the row-scoring below needs two markers, and a grid with a single
# money column would otherwise score 1 and be skipped as junk.
_LABEL_HEADERS = {
    "categoria", "categoria profesional", "categorias", "grupo", "grupos",
    "grupo profesional", "nivel", "niveles", "puesto", "denominacion",
}

# Tokens that mark a row as a salary-grid HEADER.
_HEADER_MARKERS = _MONEY_HEADERS | _LABEL_HEADERS


def _is_number(v) -> bool:
    if isinstance(v, (int, float)):
        return True
    if v is None:
        return False
    s = str(v).strip().replace(".", "").replace(",", "")
    return s.isdigit()


def _to_float(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    # Spanish decimal-comma normalization (1.652,13 → 1652.13), as registry import.
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _looks_like_hours_header(label: str) -> bool:
    # The "1742" column (a bare annual-hours number as header) holds €/hora.
    return bool(re.fullmatch(r"\d{3,4}", label))


def _year_from_sheet_name(name: str):
    m = re.search(r"(19|20)\d{2}", name)
    if m:
        return int(m.group(0))
    m = re.search(r"\b(\d{2})\b", name)  # "smi 26" → 2026
    if m:
        return 2000 + int(m.group(1))
    return None


def _find_header_row(rows: list[list]) -> int | None:
    best_idx, best_score = None, 0
    for i, row in enumerate(rows[:12]):
        score = 0
        for cell in row:
            label = _norm(cell)
            if not label:
                continue
            if label in _HEADER_MARKERS or _looks_like_hours_header(label) or label in _GROSS:
                score += 1
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx if best_score >= 2 else None


def _column_field_map(header: list[str]) -> dict:
    """col index → typed field name (gross_annual / base_salary_monthly /
    hourly_rate / extra_pay / night_plus). Unmapped numeric columns stay
    raw_values only."""
    mapping = {}
    monthly_idx, _, _ = _monthly_column(header)
    if monthly_idx is not None:
        mapping[monthly_idx] = "base_salary_monthly"
    for idx, raw in enumerate(header):
        label = _norm(raw)
        if not label or idx in mapping:
            continue
        if label in _GROSS and "gross_annual" not in mapping.values():
            mapping[idx] = "gross_annual"
        elif label in _HOURLY or _looks_like_hours_header(label):
            mapping.setdefault(idx, "hourly_rate")
        elif label in _EXTRA:
            mapping.setdefault(idx, "extra_pay")
        elif label in _NIGHT:
            mapping.setdefault(idx, "night_plus")
    return mapping


def _monthly_column(header: list[str]) -> tuple[int | None, int | None, list[str]]:
    """Which column (if any) holds a STATED monthly base salary, and the pagas
    count the header states for it (Correction-salary-01).

    Never derived: if the source labels no column as a monthly base, this
    returns None and `base_salary_monthly` stays NULL. Where a sheet offers
    several stated monthlies side by side (COEAS Navarra prints "14 pagas" and
    "12 pagas" next to each other), the 14-pagas column is taken — the Spanish
    12-plus-2 norm — and the alternative is named in a warning while staying
    verbatim in raw_values. That is a choice between two SOURCE cells, which is
    not the same thing as computing a figure the source never printed.

    @return (column index, stated pagas count, warnings)
    """
    candidates: list[tuple[int, int | None]] = []  # (col index, stated pagas count)
    for idx, raw in enumerate(header):
        label = _norm(raw)
        if not label:
            continue
        at_pagas = _MONTHLY_AT_PAGAS_RE.match(label)
        if at_pagas:
            candidates.append((idx, int(at_pagas.group(1))))
        elif label in _MONTHLY:
            stated = _PAGAS_RE.search(label)
            candidates.append((idx, int(stated.group(1)) if stated else None))

    if not candidates:
        return None, None, []

    warnings: list[str] = []
    chosen = candidates[0]
    if len(candidates) > 1:
        preferred = next((c for c in candidates if c[1] == 14), None)
        chosen = preferred or candidates[0]
        others = [f"col {i} ({_norm(header[i]) or '?'})" for i, _ in candidates if i != chosen[0]]
        warnings.append(
            f"several columns state a monthly base ({', '.join(f'col {i}' for i, _ in candidates)}); "
            f"used col {chosen[0]} ('{_norm(header[chosen[0]])}') for base_salary_monthly — "
            f"{', '.join(others)} stay verbatim in raw_values only"
        )

    idx, pagas = chosen
    if pagas is None:
        pagas, pagas_warnings = _stated_pagas(header)
        warnings.extend(pagas_warnings)
    return idx, pagas, warnings


def _stated_pagas(header: list[str]) -> tuple[int | None, list[str]]:
    """The pagas count stated ANYWHERE in the header row, when the row agrees
    with itself. Disagreeing statements ("14 pagas" and "12 pagas" both
    present) yield None — the count is then genuinely a convenio-level fact the
    sheet does not settle, and it is never guessed."""
    stated = set()
    for raw in header:
        found = _PAGAS_RE.search(_norm(raw))
        if found:
            stated.add(int(found.group(1)))
    if len(stated) == 1:
        return stated.pop(), []
    if len(stated) > 1:
        return None, [
            f"header states more than one pagas count ({sorted(stated)}) — pagas_count left NULL "
            "(not guessed); every figure stays verbatim in raw_values"
        ]
    return None, []


def _parse_sheet(name: str, rows: list[list]) -> tuple[dict | None, list[str], dict]:
    """@return (table or None, warnings, diagnostic)

    The diagnostic is what lets `salary:import` fail LOUDLY instead of
    reporting success over an empty import (Correction-salary-01, priority 2):
    `no_header` is the benign junk/notes case, while `header_but_no_rows` and
    `header_maps_to_nothing` mean a grid WAS recognized and yielded nothing.
    """
    warnings: list[str] = []
    diagnostic = {"sheet": name, "status": "ok", "typed_fields": [], "rows": 0}
    header_idx = _find_header_row(rows)
    if header_idx is None:
        diagnostic["status"] = "no_header"
        return None, warnings, diagnostic
    width = _width(rows)
    header = [_norm_or_index(rows[header_idx], i) for i in range(width)]
    field_map = _column_field_map(rows[header_idx])
    monthly_idx, pagas_count, monthly_warnings = _monthly_column(rows[header_idx])
    warnings.extend(f"sheet '{name}': {w}" for w in monthly_warnings)
    diagnostic["typed_fields"] = sorted(set(field_map.values()))
    if not field_map:
        # A header row was recognized but not one of its columns maps to a typed
        # field — every figure would land in raw_values only, which is a silent
        # non-import, not a successful one.
        diagnostic["status"] = "header_maps_to_nothing"
        warnings.append(
            f"sheet '{name}': a header row was found on row {header_idx} but NO column maps to a "
            f"typed salary field (headers: {[h for h in header if h]}) — nothing would be typed"
        )
        return None, warnings, diagnostic
    data = rows[header_idx + 1:]

    # Label columns = everything LEFT of the first money-header column (the
    # numeric grid). Robust across formats whose label is a code that *looks*
    # numeric (Cantabria "3.1") — header position, not value type, decides.
    money_cols = [c for c in range(width) if header[c] in _MONEY_HEADERS or _looks_like_hours_header(header[c])]
    if money_cols:
        first_money = min(money_cols)
    else:
        # Fallback: leftmost column whose data is mostly numeric.
        numeric_cols = []
        for c in range(width):
            nonempty = [r[c] for r in data if c < len(r) and r[c] not in (None, "")]
            if nonempty and sum(_is_number(v) for v in nonempty) / len(nonempty) > 0.5:
                numeric_cols.append(c)
        first_money = min(numeric_cols) if numeric_cols else width
    label_cols = list(range(first_money))

    out_rows = []
    for r in data:
        # A data row must carry at least one numeric value in the grid.
        if not any(c < len(r) and _is_number(r[c]) for c in range(first_money, width)):
            continue
        # Label columns → group_code (leftmost) + job_category_name (rightmost),
        # skipping genuinely empty cells (never the literal string "None").
        labels = []
        for c in label_cols:
            if c < len(r) and r[c] is not None:
                s = re.sub(r"\s+", " ", str(r[c])).strip()  # collapse embedded newlines
                s = s.strip("'\u2019\u2018\"`").strip()  # strip wrapping quotes/apostrophes (e.g. "2.1'" → "2.1")
                if s:
                    labels.append(s)
        if not labels:
            continue
        job_category_name = labels[-1]
        group_code = labels[0] if len(labels) >= 2 else (
            labels[0] if re.match(r"^\d+(\.\d+)?'?$", labels[0]) else None
        )

        gross = None
        monthly = hourly = extra = night = None
        raw_values = {}
        for c in range(width):
            if c in label_cols or c >= len(r):
                continue
            val = r[c]
            if val in (None, ""):
                continue
            label = header[c] if header[c] else f"col{c}"
            raw_values[label] = val if not isinstance(val, float) else round(val, 6)
            field = field_map.get(c)
            if field == "gross_annual":
                gross = _to_float(val)
            elif field == "base_salary_monthly":
                monthly = _to_float(val)
            elif field == "hourly_rate":
                hourly = _to_float(val)
            elif field == "extra_pay":
                extra = _to_float(val)
            elif field == "night_plus":
                night = _to_float(val)

        def _bounded(field: str, value, name_for_warning: str = job_category_name):
            # Drop (never guess-correct) a typed value the DB column can't
            # hold — the raw, verbatim figure stays in raw_values regardless
            # (unconditional above), so nothing is lost, just not force-fit
            # into a typed column it structurally cannot represent.
            if value is None or abs(value) < _FIELD_BOUNDS[field]:
                return value
            warnings.append(
                f"sheet '{name}': {field}={value!r} out of range for "
                f"'{name_for_warning}' (source spreadsheet data-quality issue, not "
                f"a mapping bug) — kept in raw_values only, not written as {field}"
            )
            return None

        gross = _bounded("gross_annual", gross)
        # NEVER derived (Correction-salary-01): the monthly figure is whatever
        # the source's own monthly column says, or NULL.
        base_monthly = _bounded("base_salary_monthly", monthly)
        extra = _bounded("extra_pay", extra)
        hourly = _bounded("hourly_rate", hourly)
        night = _bounded("night_plus", night)

        out_rows.append(
            {
                "job_category_name": job_category_name,
                "group_code": group_code,
                "gross_annual": round(gross, 2) if gross is not None else None,
                "base_salary_monthly": round(base_monthly, 2) if base_monthly is not None else None,
                "extra_pay": round(extra, 2) if extra is not None else None,
                # A typed field in its own right, stated by the source or NULL.
                # NEVER used to derive another figure.
                "pagas_count": pagas_count,
                "hourly_rate": round(hourly, 4) if hourly is not None else None,
                "night_plus": round(night, 2) if night is not None else None,
                "raw_values": raw_values,
            }
        )

    if not out_rows:
        diagnostic["status"] = "header_but_no_rows"
        warnings.append(
            f"sheet '{name}': a salary-grid header was found on row {header_idx} but NOT ONE data "
            "row carried a numeric figure in the grid — 0 rows would be imported"
        )
        return None, warnings, diagnostic
    diagnostic["rows"] = len(out_rows)
    return {
        "sheet": name,
        "year": _year_from_sheet_name(name),
        "validity_start": None,
        "validity_end": None,
        "rows": out_rows,
    }, warnings, diagnostic


def _width(rows: list[list]) -> int:
    return max((len(r) for r in rows), default=0)


def _norm_or_index(row: list, i: int) -> str:
    return _norm(row[i]) if i < len(row) else ""


def parse_salary_xlsx(xlsx_bytes: bytes) -> dict:
    """Parse a salary workbook → {tables, warnings, sheet_diagnostics}.

    One table per salary sheet (multi-year supported). Junk/notes sheets that
    have no salary-grid header are skipped and reported in warnings.

    `sheet_diagnostics` (Correction-salary-01) reports, per sheet, whether a
    grid was recognized and what it yielded, so `salary:import` can FAIL rather
    than report success when a recognized grid produced nothing:
    `ok` | `empty` | `no_header` | `header_but_no_rows` | `header_maps_to_nothing`.
    """
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), data_only=True, read_only=True)
    tables, warnings, diagnostics = [], [], []
    for name in wb.sheetnames:
        ws = wb[name]
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        if not rows or _width(rows) < 2:
            warnings.append(f"sheet '{name}' skipped (empty/too small)")
            diagnostics.append({"sheet": name, "status": "empty", "typed_fields": [], "rows": 0})
            continue
        parsed, sheet_warnings, diagnostic = _parse_sheet(name, rows)
        warnings.extend(sheet_warnings)
        diagnostics.append(diagnostic)
        if parsed is None:
            if diagnostic["status"] == "no_header":
                warnings.append(f"sheet '{name}' skipped (no salary-grid header found)")
            continue
        hdr = _find_header_row(rows)
        warnings.append(f"sheet '{name}': header on row {hdr}, {len(parsed['rows'])} category rows, year {parsed['year']}")
        tables.append(parsed)
    return {"tables": tables, "warnings": warnings, "sheet_diagnostics": diagnostics}
