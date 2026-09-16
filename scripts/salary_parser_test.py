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

print("5b. the monthly figure carries the header it was read from, verbatim")
result = parse_salary_xlsx(workbook({"2025": [
    ["CATEGORÍA", "Salario Base", "Bruto mes", "Bruto anual"],
    ["Grupo 3", 1183.34, 1771.64, 21259.75],
]}))
row = result["tables"][0]["rows"][0]
check("the label keeps the source's own casing", row["base_salary_monthly_label"], "Salario Base")
check("the typed figure is the base, not the bruto", row["base_salary_monthly"], 1183.34)
check("and the other monthly stays verbatim in raw_values", row["raw_values"]["bruto mes"], 1771.64)

print("5c. a monthly read from an 'N pagas' column is labelled as that column")
result = parse_salary_xlsx(workbook({"2026": [
    ["Grupo", "14 pagas", "12 pagas", "Hora"],
    ["Director/a", 2231.04, 2602.88, 18.33],
]}))
row = result["tables"][0]["rows"][0]
check("labelled '14 pagas', never 'salario base'", row["base_salary_monthly_label"], "14 pagas")

print("6. a multi-year block repeating a header keeps BOTH figures and types neither")
result = parse_salary_xlsx(workbook({"2024-2025": [
    ["Grupo", "2025", "14 pagas", "12 pagas", "2026", "14 pagas", "12 pagas"],
    ["Director", 30324.80, 2166.06, 2527.07, 31234.54, 2231.04, 2602.88],
]}))
row = result["tables"][0]["rows"][0]
check("no monthly is filed under the wrong year", row["base_salary_monthly"], None)
check("nor a pagas count", row["pagas_count"], None)
check("the first 14-pagas figure survives", row["raw_values"]["14 pagas"], 2166.06)
check("and so does the second, suffixed", row["raw_values"]["14 pagas (2)"], 2231.04)

print("7. sheet_diagnostics separate a notes sheet from an empty recognized grid")
result = parse_salary_xlsx(workbook({
    "2026": [["Categoría", "Salario anual"], ["Grupo I", 33491.36]],
    "Notes": [["page", "note"], ["1", "publicado en el BOG"]],
    "Vacía": [["Categoría", "Salario anual"], ["", ""]],
}))
check("a real grid", sheet_status(result, "2026"), "ok")
check("a notes sheet is benign", sheet_status(result, "Notes"), "no_header")
check("a recognized grid with no data rows is NOT benign", sheet_status(result, "Vacía"), "header_but_no_rows")
check("only the real grid became a table", [t["sheet"] for t in result["tables"]], ["2026"])

print("8. a monthly-labelled column holding an ANNUAL is refused (COEAS Andalucía's bare 'SB')")
# The real file: 'SB' and 'TOTAL' print the same 22.058,76 annual, and the true
# monthly (1.575,63) sits in the adjacent bare "14" column. 'SB' is in _MONTHLY,
# so without this invariant the annual is typed as the monthly base — 14x wrong,
# and invisible to `salary:audit-monthly` because it IS a real source cell.
result = parse_salary_xlsx(workbook({"smi 26": [
    ["Categoría", "SB", "TOTAL", 14, 12],
    ["Director/a Gerente", 22058.76, 22058.76, 1575.63, 0],
]}))
row = result["tables"][0]["rows"][0]
check("the annual is not stored as a monthly", row["base_salary_monthly"], None)
check("and carries no monthly label either", row["base_salary_monthly_label"], None)
check("the annual is still stored as the annual", row["gross_annual"], 22058.76)
check("the mislabelled figure stays verbatim", row["raw_values"]["sb"], 22058.76)
check("as does the real monthly, unread", row["raw_values"]["14"], 1575.63)
check(
    "and the refusal is reported, not silent",
    any("ANNUAL" in w for w in result["warnings"]),
    True,
)

print("8a. the column verdict governs the top-up rows the row-level test can't catch")
# The row-level test passes on 'Mediador/a' (SB 17411.25 < TOTAL 17847.05, because
# COMP. tops the row up) and would file that annual as a monthly. The column holds
# annuals on the majority of rows, so it is refused for EVERY row of the sheet.
result = parse_salary_xlsx(workbook({"smi 26": [
    ["Categoría", "SB", "COMP.", "TOTAL", 14],
    ["Director/a Gerente", 22058.76, None, 22058.76, 1575.63],
    ["Jefe/a de Departamento", 19680.81, None, 19680.81, 1405.77],
    ["Mediador/a Intercultural Educativo", 17411.25, 435.80, 17847.05, 1243.66],
]}))
rows = result["tables"][0]["rows"]
check("all three rows parsed", len(rows), 3)
check("the top-up row types NO monthly either", rows[2]["base_salary_monthly"], None)
check("nor do the equal rows", [r["base_salary_monthly"] for r in rows], [None, None, None])
check("the annuals are all still stored", [r["gross_annual"] for r in rows], [22058.76, 19680.81, 17847.05])
check("the top-up row's SB stays verbatim", rows[2]["raw_values"]["sb"], 17411.25)
check(
    "one column-level warning, not one per row",
    sum("holds ANNUAL figures in this workbook" in w for w in result["warnings"]),
    1,
)
check(
    "base_salary_monthly is dropped from the sheet's typed fields",
    "base_salary_monthly" in result["sheet_diagnostics"][0]["typed_fields"],
    False,
)

print("8b. a legitimate monthly BELOW its annual is unaffected by either term")
result = parse_salary_xlsx(workbook({"2026": [
    ["Categoría", "Salario base", "Salario anual"],
    ["Grupo I", 2232.75, 33491.36],
]}))
row = result["tables"][0]["rows"][0]
check("still stored", row["base_salary_monthly"], 2232.75)
check("still labelled", row["base_salary_monthly_label"], "Salario base")

print("8c. the row-level term still catches a lone bad row a genuine monthly column contains")
# The column is a real monthly column (majority of rows are below the annual), so
# the column verdict does NOT fire — one anomalous row must still be refused on its
# own, which is why both terms exist rather than just the column one.
result = parse_salary_xlsx(workbook({"2026": [
    ["Categoría", "Salario base", "Bruto anual"],
    ["Grupo I", 2232.75, 33491.36],
    ["Grupo II", 2100.00, 31000.00],
    ["Grupo III", 18000.00, 18000.00],
]}))
rows = result["tables"][0]["rows"]
check("the good rows keep their monthly", [r["base_salary_monthly"] for r in rows[:2]], [2232.75, 2100.00])
check("the anomalous row does not", rows[2]["base_salary_monthly"], None)
check("and it loses its label with it", rows[2]["base_salary_monthly_label"], None)
check("its figure survives verbatim", rows[2]["raw_values"]["salario base"], 18000.00)
check(
    "reported per row, naming the category",
    any("Grupo III" in w and "not below" in w for w in result["warnings"]),
    True,
)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
    sys.exit(1)
print("All salary parser contract checks passed.")
