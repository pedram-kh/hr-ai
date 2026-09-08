"""Salary parser contract test — Correction-salary-01.

Pins the rule that made the correction necessary: a salary figure is READ from a
source cell or it is not stored at all (ADR-0006). The parser used to compute
`base_salary_monthly = gross_annual / 14` and assert `num_payments = 14` for
every table, which told convenio 15 (Gestores Información Gipuzkoa, whose annual
is 15 × its monthly base) a monthly of 2.392,24 € where its own gazette prints
2.232,75 €.

Builds the workbooks in memory (no S3, no DB, no network) and asserts:
  1. an annual-only sheet stores NO monthly — it is never derived;
  2. a sheet with a monthly column stores THAT figure, verbatim;
  3. `pagas_count` comes only from a header that states it, and is never used
     to derive a figure;
  4. a header stating two different pagas counts leaves `pagas_count` NULL;
  5. `sheet_diagnostics` distinguishes a benign notes sheet from a recognized
     grid that yielded nothing (what makes `salary:import` fail loudly).

Run:
    python scripts/salary_parser_test.py        # from hr-ai/
    docker exec hr_ai python scripts/salary_parser_test.py
"""

from __future__ import annotations

import io
import sys

import openpyxl

sys.path.insert(0, "/app")
sys.path.insert(0, ".")

from app.salary import parse_salary_xlsx  # noqa: E402

FAILURES: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual == expected:
        print(f"  ok   {label}: {actual!r}")
    else:
        print(f"  FAIL {label}: expected {expected!r}, got {actual!r}")
        FAILURES.append(label)


def workbook(sheets: dict[str, list[list]]) -> bytes:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for row in rows:
            ws.append(row)
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def sheet_status(result: dict, name: str) -> str | None:
    for diagnostic in result["sheet_diagnostics"]:
        if diagnostic["sheet"] == name:
            return diagnostic["status"]
    return None


print("1. annual only → monthly is NULL, never gross/14")
result = parse_salary_xlsx(workbook({"2026": [
    ["Categoría", "Salario anual"],
    ["Grupo I", 33491.36],
    ["Grupo II", 28000.00],
]}))
row = result["tables"][0]["rows"][0]
check("gross_annual", row["gross_annual"], 33491.36)
check("base_salary_monthly is NOT derived", row["base_salary_monthly"], None)
check("pagas_count is not assumed", row["pagas_count"], None)
check("the annual is still kept verbatim", row["raw_values"]["salario anual"], 33491.36)

print("2. a monthly column → that figure is stored, and the annual is NOT divided")
result = parse_salary_xlsx(workbook({"2026": [
    ["Categoría", "Salario base", "Salario anual"],
    ["Grupo I", 2232.75, 33491.36],
]}))
row = result["tables"][0]["rows"][0]
check("base_salary_monthly is the source cell", row["base_salary_monthly"], 2232.75)
check("NOT gross/14 (2392.24)", row["base_salary_monthly"] == round(33491.36 / 14, 2), False)
check("gross_annual", row["gross_annual"], 33491.36)

print("3. pagas_count is read from the header that states it")
result = parse_salary_xlsx(workbook({"2026": [
    ["Categoría", "Bruto anual", "14 pagas"],
    ["Grupo I", 31234.54, 2231.04],
]}))
row = result["tables"][0]["rows"][0]
check("pagas_count", row["pagas_count"], 14)
check("monthly is the 14-pagas cell", row["base_salary_monthly"], 2231.04)
check("gross_annual is untouched", row["gross_annual"], 31234.54)

print("4. two stated monthlies → the 14-pagas one is used, the other stays raw")
result = parse_salary_xlsx(workbook({"2026": [
    ["Categoría", "Bruto anual", "14 pagas", "12 pagas"],
    ["Grupo I", 31234.54, 2231.04, 2602.88],
]}))
row = result["tables"][0]["rows"][0]
check("monthly", row["base_salary_monthly"], 2231.04)
check("pagas_count", row["pagas_count"], 14)
check("the 12-pagas figure is not lost", row["raw_values"]["12 pagas"], 2602.88)

print("5. a header stating two counts and no monthly column → pagas_count NULL")
result = parse_salary_xlsx(workbook({"2026": [
    ["Categoría", "Bruto anual", "Bruto/mes 14 pagas", "Bruto/mes 12 pagas", "Salario base"],
    ["Grupo I", 31234.54, 2231.04, 2602.88, 1900.00],
]}))
row = result["tables"][0]["rows"][0]
check("monthly is the base column", row["base_salary_monthly"], 1900.00)
check("pagas_count is not guessed between 14 and 12", row["pagas_count"], None)

print("6. sheet_diagnostics separate a notes sheet from an empty recognized grid")
result = parse_salary_xlsx(workbook({
    "2026": [["Categoría", "Salario anual"], ["Grupo I", 33491.36]],
    "Notes": [["page", "note"], ["1", "publicado en el BOG"]],
    "Vacía": [["Categoría", "Salario anual"], ["", ""]],
}))
check("a real grid", sheet_status(result, "2026"), "ok")
check("a notes sheet is benign", sheet_status(result, "Notes"), "no_header")
check("a recognized grid with no data rows is NOT benign", sheet_status(result, "Vacía"), "header_but_no_rows")
check("only the real grid became a table", [t["sheet"] for t in result["tables"]], ["2026"])

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
    sys.exit(1)
print("All salary parser contract checks passed.")
