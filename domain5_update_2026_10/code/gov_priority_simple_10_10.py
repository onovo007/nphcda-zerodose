"""Simplified government workbook (10.10.2026): LGA priority lists ranked by zero-dose burden.

Format of the research workbook (gov_workbook_10_10.py), content of the simplified 7.8.2026 priority lists: LGAs ranked by
the estimated number of zero-dose children (Method 1, the headline estimate), with priority band, archetype/contextual
profile, dominant barrier and candidate components. No priority probabilities or uncertainty intervals.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import DataBarRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "government_workbook"
EV = ROOT.parents[1] / "nphcda-zerodose-git" / "data" / "sample" / "two_methods" / "evidence"
NAVY, STEEL, M2F = "1F3B57", "2E6E8E", "FBE5D6"
BAND_FILL = {"A: first 50% of the burden": "F6C9AE", "B: 50% to 80%": "FBE7B0", "C: last 20%": "DCE9F5"}
thin = Side(style="thin", color="D5DCE3")
M1 = "Method 1: Bayesian hierarchical model with DHIS2-calibrated LGA allocation"
M2 = "Method 2: Bayesian small-area estimation (SAE)"
PLAB = {1: "Deprived northern rural", 2: "Low maternal care, high conflict exposure", 3: "Geographically isolated",
        4: "Near-average", 5: "Relatively advantaged"}
PROF = "Archetype/Contextual profile"
COMP = "Candidate components (local validation needed)"

lg = pd.read_csv(RES / "lga_both_methods_774.csv")
lg = lg.loc[:, ~lg.columns.duplicated()]
zn = pd.read_csv(RES / "national_zone_both_methods.csv")
par = pd.read_csv(RES / "pareto_scenarios.csv")
tk = pd.read_csv(RES / "topk_shares.csv")
ar = pd.read_csv(RES / "archetype_both_methods.csv")
R = json.loads((RES / "results.json").read_text())
bar = pd.read_csv(EV / "lga_barriers_774.csv")
BMAT = pd.read_csv(EV / "barrier_matrix.csv")

lg["profile"] = lg["archetype"].map(PLAB)
lg = lg.merge(bar[["lga_uid", "dominant_barrier", "candidate_components"]], on="lga_uid", how="left")
lg["dominant_barrier"] = lg["dominant_barrier"].str.replace("Programme (dropout)", "DTP1-3 dropout", regex=False) \
    .replace({"None above +0.5 SD": "None above average"})

# rank by Method 1 burden (773 LGAs); Guzamala (no Penta1 data 2021-2024) listed last, unranked
L = lg.sort_values(["m1_children", "m2_children"], ascending=[False, False], na_position="last").reset_index(drop=True)
est = L["m1_children"].notna()
L["rank"] = pd.Series(np.where(est, np.arange(1, len(L) + 1), np.nan))
L["cum"] = np.where(est, L["m1_children"].cumsum() / L.loc[est, "m1_children"].sum() * 100, np.nan)
L["band"] = np.where(~est, "", np.where(L["cum"] <= 50, "A: first 50% of the burden",
                                        np.where(L["cum"] <= 80, "B: 50% to 80%", "C: last 20%")))
# make the band boundary inclusive of the LGA that crosses the threshold (matches 151 / 342)
for cut, lab in ((50, "A: first 50% of the burden"), (80, "B: 50% to 80%")):
    first_over = int((L["cum"] > cut).idxmax())
    L.loc[first_over, "band"] = lab
N50, N60, N80 = R["m1"]["pareto"]["50"], R["m1"]["pareto"]["60"], R["m1"]["pareto"]["80"]
assert (L["band"] == "A: first 50% of the burden").sum() == N50, (L["band"] == "A: first 50% of the burden").sum()
assert (L["band"].isin(["A: first 50% of the burden", "B: 50% to 80%"])).sum() == N80
TOP = 155
top_share = L.loc[L["rank"] <= TOP, "m1_children"].sum() / L.loc[est, "m1_children"].sum() * 100

LIST_COLS = {"rank": "Burden rank", "state": "State", "lga_clean": "LGA", "zone": "Zone",
             "m1_children": "Zero-dose children (est)", "m1_children_lo95": "Lower 95% CrI", "m1_children_hi95": "Upper 95% CrI",
             "m1_rate": "Zero-dose rate (%)", "cum": "Cumulative % of burden",
             "band": "Priority band", "profile": PROF, "dominant_barrier": "Dominant barrier", "candidate_components": COMP,
             "m2_children": "Method 2 (SAE) zero-dose children", "m2_children_lo95": "Method 2 lower 95% CrI",
             "m2_children_hi95": "Method 2 upper 95% CrI", "m2_rank": "Method 2 (SAE) rank"}
Lw = L[list(LIST_COLS)].rename(columns=LIST_COLS)
for c in ("Zero-dose children (est)", "Lower 95% CrI", "Upper 95% CrI", "Method 2 (SAE) zero-dose children",
          "Method 2 lower 95% CrI", "Method 2 upper 95% CrI"):
    Lw[c] = Lw[c].round(0)
Lw.loc[~est.values, "Priority band"] = "Not ranked: no Penta1 doses reported 2021-2024"
LFMT = {"Burden rank": "0", "Zero-dose children (est)": "#,##0", "Lower 95% CrI": "#,##0", "Upper 95% CrI": "#,##0",
        "Zero-dose rate (%)": "0.0", "Cumulative % of burden": "0.0", "Method 2 (SAE) zero-dose children": "#,##0",
        "Method 2 lower 95% CrI": "#,##0", "Method 2 upper 95% CrI": "#,##0", "Method 2 (SAE) rank": "0"}
LWID = {"Burden rank": 9, "State": 13, "LGA": 22, "Zone": 14, "Zero-dose children (est)": 14, "Zero-dose rate (%)": 11,
        "Cumulative % of burden": 12, "Priority band": 24, PROF: 40, "Dominant barrier": 18, COMP: 72,
        "Lower 95% CrI": 11, "Upper 95% CrI": 11, "Method 2 (SAE) zero-dose children": 14, "Method 2 lower 95% CrI": 12,
        "Method 2 upper 95% CrI": 12, "Method 2 (SAE) rank": 11}

wb = Workbook()


def header(ws, row, cols):
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=row, column=j, value=c)
        cell.font = Font(bold=True, color="FFFFFF", size=10); cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center"); cell.border = Border(bottom=thin)


def title(ws, text, sub=None):
    ws["A1"] = text; ws["A1"].font = Font(bold=True, size=15, color=NAVY)
    if sub:
        ws["A2"] = sub; ws["A2"].font = Font(italic=True, size=9.5, color="5B6B79")


def write_table(ws, df, start_row, fmts=None, name=None, widths=None, wrap=(), fills=None, height=32):
    header(ws, start_row, list(df.columns))
    ws.row_dimensions[start_row].height = height
    for i, row in enumerate(df.itertuples(index=False), start_row + 1):
        for j, v in enumerate(row, 1):
            v = None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
            c = ws.cell(row=i, column=j, value=v)
            col = df.columns[j - 1]
            c.font = Font(size=10); c.border = Border(bottom=thin)
            if fmts and col in fmts:
                c.number_format = fmts[col]
            if col in wrap:
                c.alignment = Alignment(wrap_text=True, vertical="top")
            if fills and col in fills and v in fills[col]:
                c.fill = PatternFill("solid", fgColor=fills[col][v])
    for j, col in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(j)].width = (widths or {}).get(col, max(11, min(40, len(str(col)) * 0.9)))
    if name:
        t = Table(displayName=name, ref=f"A{start_row}:{get_column_letter(len(df.columns))}{start_row + len(df)}")
        t.tableStyleInfo = TableStyleInfo(name="TableStyleLight1", showRowStripes=True)
        ws.add_table(t)
    return start_row + len(df)


def page(ws, freeze=None, title_row=None):
    ws.sheet_view.zoomScale = 90
    if freeze:
        ws.freeze_panes = freeze  # set once per sheet
    ws.page_setup.orientation = "landscape"; ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth = 1; ws.page_setup.fitToHeight = 0; ws.sheet_properties.pageSetUpPr.fitToPage = True
    if title_row:
        ws.print_title_rows = f"{title_row}:{title_row}"


def note(ws, row, text):
    c = ws.cell(row=row, column=1, value=text); c.font = Font(italic=True, size=9, color="5B6B79")


# ---------------------------------------------------------------- About
ws = wb.active; ws.title = "About"
title(ws, "Zero-dose children in Nigeria, 2026: LGA priority lists",
      "National Primary Health Care Development Agency (NPHCDA). Model estimates prepared by the CIDRE and Quantium Insights LLC consortium, with UNICEF and Gavi.")
rows = [
    ("What this workbook contains", "Every local government area (LGA) ranked by its estimated number of zero-dose children in 2026 (children aged 12-23 months "
     "who have not received the first dose of the pentavalent vaccine, Penta1/DTP1), with the archetype (contextual profile) of each LGA, its main "
     "barrier and the candidate intervention components to consider."),
    ("How LGAs are ranked", "By burden: the number of zero-dose children. A large LGA can rank above a smaller one with a higher rate, because it holds more "
     "unvaccinated children. This is how high-burden areas are prioritized for investment."),
    ("Method 1", M1 + ". A Bayesian hierarchical Beta regression estimates each state's zero-dose rate from the Nigeria Demographic and Health "
     "Surveys (NDHS) 2008, 2013, 2018 and 2023-24, with a DHIS2 Penta1 trend term, and forecasts it to 2026. Each state's rate is then distributed "
     "to its LGAs using each LGA's share of Penta1 doses in DHIS2 and its population. Estimates for 773 LGAs; Guzamala (Borno) is listed but not "
     "ranked, because it reported no Penta1 doses in 2021-2024. The ranking in this workbook uses Method 1."),
    ("Method 2", M2 + ". Each LGA has its own zero-dose rate. The population-weighted average of a state's LGA rates must agree with that state's "
     "NDHS results (allowing for survey sampling error). Differences between LGAs are informed by local conditions (maternal care, water, wealth, "
     "travel time to a facility, conflict and poverty) and by neighboring LGAs (spatial smoothing). Estimates for all 774 LGAs, with full uncertainty."),
    ("Reading the two methods", "Both methods agree closely on national, zone and state totals. They differ mainly in how zero-dose children are "
     "distributed among the LGAs of a state. LGAs that rank high under both methods are the most robust priorities."),
    ("95% credible interval (CrI)", "The range within which the true number of zero-dose children is expected to lie with 95% probability under each "
     "model. Neither interval includes error in the population figures."),
    ("Priority band", "A = the highest-burden LGAs that together hold the first 50% of zero-dose children; B = the next LGAs, up to 80%; C = the remaining 20%."),
    ("Archetype/Contextual profile", "One of five groups of LGAs with similar social and service conditions (clustering of 15 indicators on poverty, "
     "education, maternal care, nutrition, access, conflict and DTP1-3 dropout): deprived northern rural; low maternal care, high conflict exposure; "
     "geographically isolated; near-average; relatively advantaged."),
    ("Dominant barrier", "The LGA's strongest contextual barrier among socioeconomic, maternal care, nutrition, access, insecurity and DTP1-3 dropout."),
    ("Candidate components", "Intervention components matched to the LGA's flagged barriers, on top of the core of fixed-site and outreach routine "
     "immunization with microplanning. They are starting points to be validated locally during microplanning (see 'Profiles and packages')."),
    ("Sheets", f"About | National and zone | All ranked LGAs | Top {TOP} LGAs (20%) | Top {N80} LGAs (80% of burden) | Burden concentration | Profiles and packages"),
    ("Status", "Model estimates for programme planning. Figures are estimates, not counts, and are best used as a relative priority ranking."),
]
for i, (k, v) in enumerate(rows, 4):
    ws.cell(row=i, column=1, value=k).font = Font(bold=True, color=NAVY, size=10.5)
    ws.cell(row=i, column=1).alignment = Alignment(vertical="top")
    c = ws.cell(row=i, column=2, value=v); c.alignment = Alignment(wrap_text=True, vertical="top"); c.font = Font(size=10.5)
    ws.row_dimensions[i].height = max(30, 15 * (len(v) // 120 + 1))
ws.column_dimensions["A"].width = 52; ws.column_dimensions["B"].width = 118; ws.column_dimensions["C"].width = 18
r0 = 4 + len(rows) + 1
ws.cell(row=r0, column=1, value="Headline figures").font = Font(bold=True, size=12, color=NAVY)
m1, m2 = R["m1"], R["m2"]
heads = [("LGAs estimated", m1["lgas"], m2["lgas"], "#,##0"),
         ("Zero-dose children, 2026", m1["national"], m2["national"], "#,##0"),
         ("95% CrI, lower", m1["national_lo95"], m2["national_lo95"], "#,##0"),
         ("95% CrI, upper", m1["national_hi95"], m2["national_hi95"], "#,##0"),
         ("LGAs holding 50% of zero-dose children", m1["pareto"]["50"], m2["pareto"]["50"], "#,##0"),
         ("LGAs holding 60% of zero-dose children", m1["pareto"]["60"], m2["pareto"]["60"], "#,##0"),
         ("LGAs holding 80% of zero-dose children", m1["pareto"]["80"], m2["pareto"]["80"], "#,##0"),
         (f"Share of zero-dose children in the top {TOP} LGAs (20% of LGAs)", m1["top155_share"] / 100, m2["top155_share"] / 100, "0.0%")]
for j, h in enumerate(["Comparison of the two methods", "Method 1", "Method 2 (SAE)"], 1):
    c = ws.cell(row=r0 + 1, column=j, value=h)
    c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor=NAVY)
    c.alignment = Alignment(horizontal="left" if j == 1 else "right")
for i, (k, a, b, fm) in enumerate(heads, r0 + 2):
    ws.cell(row=i, column=1, value=k).font = Font(color=NAVY, size=10.5)
    for j, v in ((2, a), (3, b)):
        c = ws.cell(row=i, column=j, value=v); c.number_format = fm; c.alignment = Alignment(horizontal="right"); c.font = Font(size=10.5)
        c.border = Border(bottom=thin)
    ws.cell(row=i, column=1).border = Border(bottom=thin)
ws.sheet_view.showGridLines = False
page(ws)

# ---------------------------------------------------------------- National and zone
ws = wb.create_sheet("National and zone")
title(ws, "Zero-dose children by zone, 2026: Method 1 and Method 2", "Model estimates with 95% credible intervals (CrI). Shares are of each method's national total.")
z = zn.rename(columns={"area": "Area", "lgas_total": "LGAs", "cohort_12_23m": "Children 12-23 months",
                       "m1_children": "M1 zero-dose children", "m1_lo95": "M1 lower 95% CrI", "m1_hi95": "M1 upper 95% CrI",
                       "m1_rate": "M1 zero-dose rate (%)", "m1_share": "M1 share of national (%)",
                       "m2_children": "M2 zero-dose children", "m2_lo95": "M2 lower 95% CrI", "m2_hi95": "M2 upper 95% CrI",
                       "m2_rate": "M2 zero-dose rate (%)", "m2_share": "M2 share of national (%)"})
z = z[["Area", "LGAs", "Children 12-23 months", "M1 zero-dose children", "M1 lower 95% CrI", "M1 upper 95% CrI", "M1 zero-dose rate (%)",
       "M1 share of national (%)", "M2 zero-dose children", "M2 lower 95% CrI", "M2 upper 95% CrI", "M2 zero-dose rate (%)",
       "M2 share of national (%)"]]
zf = {c: "#,##0" for c in z.columns if "children" in c.lower() or "CrI" in c}
zf.update({c: "0.0" for c in z.columns if "(%)" in c})
end = write_table(ws, z, 4, zf, "ZoneTable", widths={"Area": 16, **{c: 13 for c in z.columns[1:]}})
for i in range(5, end + 1):
    for j, col in enumerate(z.columns, 1):
        if col.startswith("M2"):
            ws.cell(row=i, column=j).fill = PatternFill("solid", fgColor=M2F)
for col in range(1, len(z.columns) + 1):
    ws.cell(row=5, column=col).font = Font(bold=True, size=10)
ws.conditional_formatting.add(f"D6:D{end}", DataBarRule(start_type="num", start_value=0, end_type="max", color="ED7D31"))
page(ws, "B5", 4)

# ---------------------------------------------------------------- Ranked lists
BAND = {"Priority band": BAND_FILL}


def ranked_sheet(name, df, ttl, sub, tname):
    ws = wb.create_sheet(name)
    title(ws, ttl, sub)
    end = write_table(ws, df, 4, LFMT, tname, LWID, wrap=(COMP,), fills=BAND)
    ws.conditional_formatting.add(f"E5:E{end}", DataBarRule(start_type="num", start_value=0, end_type="max", color="ED7D31"))
    page(ws, "D5", 4)


ranked_sheet("All ranked LGAs", Lw, "All LGAs ranked by estimated zero-dose children, 2026",
             "Rank 1 = most zero-dose children. Use the filter arrows to select a state, zone, band or archetype.", "TableAllLGAs")
ranked_sheet(f"Top {TOP} LGAs (20%)", Lw[Lw["Burden rank"] <= TOP],
             f"The {TOP} highest-burden LGAs (top 20% of LGAs): {top_share:.0f}% of all zero-dose children",
             "Ranked by estimated zero-dose children, 2026.", "TableTopTwentyPct")
ranked_sheet(f"Top {N80} LGAs (80%)", Lw[Lw["Burden rank"] <= N80],
             f"The {N80} highest-burden LGAs: together 80% of all zero-dose children",
             f"Bands A ({N50} LGAs, first 50%) and B (to 80%).", "TableTopEightyPct")

# ---------------------------------------------------------------- Burden concentration
ws = wb.create_sheet("Burden concentration")
title(ws, "How many LGAs hold 50%, 60% and 80% of zero-dose children", "LGAs ranked by each method's estimated zero-dose children. M1 = Method 1; M2 = Method 2 (SAE).")
pp = par[["share_of_burden", "m1_lgas", "m1_pct_of_lgas", "m2_lgas", "m2_pct_of_lgas"]].rename(columns={
    "share_of_burden": "Share of zero-dose children (%)", "m1_lgas": "M1 LGAs needed", "m1_pct_of_lgas": "M1 % of all LGAs",
    "m2_lgas": "M2 LGAs needed", "m2_pct_of_lgas": "M2 % of all LGAs"})
end = write_table(ws, pp, 4, {"Share of zero-dose children (%)": "0", "M1 LGAs needed": "0", "M1 % of all LGAs": "0.0",
                              "M2 LGAs needed": "0", "M2 % of all LGAs": "0.0"}, "ParetoTable",
                  widths={c: 16 for c in pp.columns})
ws.cell(row=end + 3, column=1, value="Share of zero-dose children held by the top-ranked LGAs").font = Font(bold=True, size=12, color=NAVY)
tt = tk[["top_k", "m1_share", "m1_lo95", "m1_hi95", "m2_share", "m2_lo95", "m2_hi95"]].rename(columns={
    "top_k": "Top-ranked LGAs", "m1_share": "M1 share (%)", "m1_lo95": "M1 lower 95% CrI", "m1_hi95": "M1 upper 95% CrI",
    "m2_share": "M2 share (%)", "m2_lo95": "M2 lower 95% CrI", "m2_hi95": "M2 upper 95% CrI"})
write_table(ws, tt, end + 4, {c: "0.0" for c in tt.columns if c != "Top-ranked LGAs"}, "TopKTable", widths={c: 16 for c in tt.columns})
note(ws, end + 6 + len(tt), "Reading: a small share of LGAs holds most zero-dose children, so starting with band A reaches half of them.")
page(ws)

# ---------------------------------------------------------------- Profiles and packages
ws = wb.create_sheet("Profiles and packages")
title(ws, "Archetypes (contextual profiles), their zero-dose burden and the barrier-to-intervention mapping",
      "Five groups of LGAs from clustering of 15 contextual indicators. Labels describe only what the indicators measure.")
A = ar.copy()
A["Archetype/Contextual profile"] = A["archetype"].map(PLAB)
A = A.rename(columns={"archetype": "No.", "lgas": "LGAs", "cohort_share": "Share of children 12-23 months (%)",
                      "m1_children": "Zero-dose children (est)", "m1_share": "Share of zero-dose children (%)",
                      "m1_mean_rate": "Mean LGA zero-dose rate (%)"})
SIG = {1: "Highest undernutrition; lowest maternal education and facility delivery; high DTP1-3 dropout",
       2: "Highest conflict events and fatalities; weak antenatal and delivery care; high dropout",
       3: "Very long travel time to care; low improved water",
       4: "Near the national average on every indicator",
       5: "Highest wealth, education and facility delivery; moderate conflict events"}
A["What sets it apart"] = A["No."].map(SIG)
A = A[["No.", "Archetype/Contextual profile", "What sets it apart", "LGAs", "Share of children 12-23 months (%)", "Zero-dose children (est)",
       "Share of zero-dose children (%)", "Mean LGA zero-dose rate (%)"]]
end = write_table(ws, A, 4, {"Share of children 12-23 months (%)": "0.0", "Zero-dose children (est)": "#,##0",
                             "Share of zero-dose children (%)": "0.0", "Mean LGA zero-dose rate (%)": "0.0"}, "ProfileTable",
                  {"No.": 6, "Archetype/Contextual profile": 34, "What sets it apart": 44}, wrap=("What sets it apart",))
r = end + 3
ws.cell(row=r - 1, column=1, value="Barrier-to-intervention mapping (applied to each LGA; candidate components to validate locally)").font = Font(bold=True, size=12, color=NAVY)
B = BMAT[["barrier", "rule", "components", "evidence", "local_validation"]].rename(columns={
    "barrier": "Barrier", "rule": "When the barrier is flagged", "components": "Candidate components",
    "evidence": "Evidence or guideline", "local_validation": "Local validation required"})
B["Barrier"] = B["Barrier"].str.replace("Programme (dropout)", "DTP1-3 dropout", regex=False)
end = write_table(ws, B, r, None, "BarrierTable", wrap=tuple(B.columns), height=30)
for j, w in enumerate([24, 34, 44, 40, 36, 18, 18, 18], 1):
    ws.column_dimensions[get_column_letter(j)].width = w
note(ws, end + 2, "A common core of fixed-site and outreach routine immunization with microplanning applies to every LGA. Population mobility is not "
                  "measured and is never assigned from these data. Packages set by state and refined with each LGA's flagged barriers fit the data "
                  "better than one national package.")
page(ws)

out = OUT / "NPHCDA_LGA_Priority_Lists_Simplified_10.10.2026.xlsx"
wb.save(out)
print(out, "| top155 share", round(top_share, 1), "| bands", L["band"].value_counts().to_dict())
