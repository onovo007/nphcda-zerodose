"""Build the Domain 5 two-method version of the NPHCDA consolidated report.

Reads the 10.6.2026 report (never modified), updates Domain 5 text, tables and figures to the canonical results,
adds the Method 2 (Bayesian small-area estimation) and comparison subsections and the two-method validation,
and writes a new .docx in report_and_deck/. Fields (contents, lists of tables and figures) are refreshed and a PDF
exported by update_report_fields.ps1 (Word).
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches
from PIL import Image
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parents[0].parent / "Gavi Modeling Overview" / "Final_NPHCDA_Consolidated_Report_UPDATED_10.6.2026.docx"
OUT = ROOT / "report_and_deck" / "Final_NPHCDA_Consolidated_Report_Domain5_Two_Methods_10.9.2026.docx"
FIG, DFIG, RES = ROOT / "figures", ROOT / "figures" / "deck", ROOT / "results"
NB1 = ROOT / "notebooks" / "outputs_Method1_Bayesian_Hierarchical_Model"

M1N = "Method 1: Bayesian hierarchical model with DHIS2-calibrated LGA allocation"
M2N = "Method 2: Bayesian small-area estimation (SAE)"

R = json.loads((RES / "results.json").read_text())
CAPS = json.loads((FIG / "captions.json").read_text())
L = pd.read_csv(RES / "lga_both_methods_774.csv")
ST = pd.read_csv(RES / "state_both_methods.csv")
ZN = pd.read_csv(RES / "national_zone_both_methods.csv")
VAL = pd.read_csv(RES / "validation_both_methods.csv")
PAR = pd.read_csv(RES / "pareto_scenarios.csv")
TOPK = pd.read_csv(RES / "topk_shares.csv")
ARCH = pd.read_csv(RES / "archetype_both_methods.csv")
ZV = pd.read_csv(RES / "zone_validation_nmdhs.csv")
COV = pd.read_csv(FIG / "data" / "M2_06_covariate_effects.csv")
SF = pd.read_csv(NB1 / "method1_state_forecasts_2026_2028.csv")
m1, m2 = R["m1"], R["m2"]
nat = ZN[ZN.area == "National"].iloc[0]


def k(x):
    return f"{x:,.0f}"


def v(method, check):
    r = VAL[(VAL.method == method) & (VAL.check.str.startswith(check))]
    assert len(r) == 1
    return r.iloc[0]


v1n, v2n = v("Method 1", "NmDHS"), v("Method 2 (SAE)", "NmDHS")
v1i, v2i = v("Method 1", "IHME DTP1 2018, LGA overall"), v("Method 2 (SAE)", "IHME DTP1 2018, LGA overall")
v1w, v2w = v("Method 1", "IHME DTP1 2018, LGA within"), v("Method 2 (SAE)", "IHME DTP1 2018, LGA within")
P = {int(r.share_of_burden): r for r in PAR.itertuples()}
T155 = TOPK[TOPK.top_k == 155].iloc[0]
A = {int(r.archetype): r for r in ARCH.itertuples()}
a12_m1 = A[1].m1_children + A[2].m1_children
a12_s1, a12_s2 = A[1].m1_share + A[2].m1_share, A[1].m2_share + A[2].m2_share
rs_state = spearmanr(ST.m1_children, ST.m2_children).statistic
ZORDER = ["North West", "North East", "North Central", "South West", "South South", "South East"]

# ------------------------------------------------------------------ helpers (lxml level)


def _runs(el):
    return [r for r in el.iter(qn("w:r")) if r.find(qn("w:t")) is not None]


def _is_bold(r):
    rpr = r.find(qn("w:rPr"))
    if rpr is None:
        return False
    b = rpr.find(qn("w:b"))
    return b is not None and b.get(qn("w:val")) not in ("0", "false")


def _mk_run(tmpl, text, bold=None):
    r = copy.deepcopy(tmpl)
    for ch in list(r):
        if ch.tag != qn("w:rPr"):
            r.remove(ch)
    parts = text.split("\n")
    for i, part in enumerate(parts):
        if i:
            r.append(r.makeelement(qn("w:br"), {}))
        t = r.makeelement(qn("w:t"), {})
        t.text = part
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        r.append(t)
    if bold is not None:
        rpr = r.find(qn("w:rPr"))
        if rpr is None:
            rpr = r.makeelement(qn("w:rPr"), {}); r.insert(0, rpr)
        for tag in ("w:b", "w:bCs"):
            b = rpr.find(qn(tag))
            if b is not None:
                rpr.remove(b)
        if bold:
            rpr.insert(0, rpr.makeelement(qn("w:b"), {}))
    return r


def set_p(p_el, runs, tmpl_run=None):
    """Replace a paragraph's text. runs: str | [(text, bold|None)]. Keeps pPr and run formatting."""
    if isinstance(runs, str):
        runs = [(runs, None)]
    rs = _runs(p_el)
    if tmpl_run is None and not rs:
        raise ValueError("no run template")
    first = tmpl_run if tmpl_run is not None else rs[0]
    bold_t = next((r for r in rs if _is_bold(r)), None)
    reg_t = next((r for r in rs if not _is_bold(r)), None)
    first, bold_t, reg_t = [copy.deepcopy(x) if x is not None else None for x in (first, bold_t, reg_t)]
    for ch in list(p_el):
        if ch.tag in (qn("w:r"), qn("w:hyperlink"), qn("w:ins"), qn("w:del"), qn("w:smartTag")):
            p_el.remove(ch)
    for text, b in runs:
        tmpl = first if b is None else (bold_t if b else reg_t)
        if tmpl is None:
            tmpl = first
        p_el.append(_mk_run(tmpl, text, b))


def set_cell(tc, text, bold=None):
    ps = tc.findall(qn("w:p"))
    for p in ps[1:]:
        tc.remove(p)
    rs = _runs(ps[0])
    tmpl = rs[0] if rs else None
    if tmpl is None:
        tmpl = ps[0].makeelement(qn("w:r"), {})
    set_p(ps[0], [(text, bold)], tmpl_run=tmpl)


def fill_table(tbl, rows, header=None):
    """Keep header row (optionally retitled) and use row 1 as template for body rows."""
    trs = tbl.findall(qn("w:tr"))
    tmpl = copy.deepcopy(trs[1])
    for tr in trs[1:]:
        tbl.remove(tr)
    if header:
        for tc, h in zip(trs[0].findall(qn("w:tc")), header):
            set_cell(tc, h)
    for row in rows:
        tr = copy.deepcopy(tmpl)
        for tc, val in zip(tr.findall(qn("w:tc")), row):
            set_cell(tc, str(val))
        tbl.append(tr)


def make_table(template_tbl, rows, widths_in):
    """New table cloned from a template (style, fonts); rows[0] = header."""
    t = copy.deepcopy(template_tbl)
    grid = t.find(qn("w:tblGrid"))
    for g in list(grid):
        grid.remove(g)
    tw = int(sum(widths_in) * 1440)
    tblw = t.find(qn("w:tblPr")).find(qn("w:tblW"))
    if tblw is not None:
        tblw.set(qn("w:w"), str(tw)); tblw.set(qn("w:type"), "dxa")
    for w in widths_in:
        grid.append(grid.makeelement(qn("w:gridCol"), {qn("w:w"): str(int(w * 1440))}))
    look = t.find(qn("w:tblPr")).find(qn("w:tblLook"))
    if look is not None:
        look.set(qn("w:firstColumn"), "0"); look.set(qn("w:val"), "0420")
    for cnf in list(t.iter(qn("w:cnfStyle"))):
        cnf.getparent().remove(cnf)
    trs = t.findall(qn("w:tr"))
    htr, btr = copy.deepcopy(trs[0]), copy.deepcopy(trs[1])
    for tr in trs:
        t.remove(tr)
    for i, row in enumerate(rows):
        base = htr if i == 0 else btr
        tr = copy.deepcopy(base)
        tcs = tr.findall(qn("w:tc"))
        tc0 = copy.deepcopy(tcs[0])
        tc1 = copy.deepcopy(tcs[1] if len(tcs) > 1 else tcs[0])
        for tc in tcs:
            tr.remove(tc)
        for j, (val, w) in enumerate(zip(row, widths_in)):
            tc = copy.deepcopy(tc0 if j == 0 else tc1)
            tcpr = tc.find(qn("w:tcPr"))
            if tcpr is not None:
                tcw = tcpr.find(qn("w:tcW"))
                if tcw is not None:
                    tcw.set(qn("w:w"), str(int(w * 1440))); tcw.set(qn("w:type"), "dxa")
                gs = tcpr.find(qn("w:gridSpan"))
                if gs is not None:
                    tcpr.remove(gs)
            set_cell(tc, str(val))
            tr.append(tc)
        t.append(tr)
    return t


def replace_image(doc, p_el, png, width_in=None):
    blip = next(p_el.iter(qn("a:blip")))
    rid, _ = doc.part.get_or_add_image(str(png))
    blip.set(qn("r:embed"), rid)
    ext = next(p_el.iter(qn("wp:extent")))
    cx = int(width_in * 914400) if width_in else int(ext.get("cx"))
    iw, ih = Image.open(png).size
    cy = int(cx * ih / iw)
    ext.set("cx", str(cx)); ext.set("cy", str(cy))
    for e in p_el.iter(qn("a:ext")):
        if e.get("cx") is not None:
            e.set("cx", str(cx)); e.set("cy", str(cy))


class Cursor:
    """Insert clones of template paragraphs/tables after an anchor element."""

    def __init__(self, doc, anchor, T):
        self.doc, self.cur, self.T = doc, anchor, T

    def _put(self, el):
        self.cur.addnext(el); self.cur = el
        return el

    def para(self, kind, runs):
        p = copy.deepcopy(self.T[kind])
        set_p(p, runs)
        return self._put(p)

    def figure(self, png, caption, width_in=6.27):
        p = copy.deepcopy(self.T["imgp"])
        for ch in list(p):
            if ch.tag != qn("w:pPr"):
                p.remove(ch)
        self._put(p)
        from docx.text.paragraph import Paragraph
        Paragraph(p, self.doc._body).add_run().add_picture(str(png), width=Inches(width_in))
        self.para("figcap", caption)

    def table(self, caption, rows, widths, note=None):
        self.para("tabcap", caption)
        self._put(make_table(self.T["tbl"], rows, widths))
        if note:
            self.para("note", note)


# ------------------------------------------------------------------ build
doc = Document(str(SRC))
Pp = list(doc.paragraphs)
Tt = list(doc.tables)
assert Pp[210].text.startswith("5.5 Domain 5") and Pp[286].text.startswith("5.9 External")
T = {"body": copy.deepcopy(Pp[211]._p), "note": copy.deepcopy(Pp[241]._p), "tabcap": copy.deepcopy(Pp[212]._p),
     "figcap": copy.deepcopy(Pp[216]._p), "h2": copy.deepcopy(Pp[244]._p), "imgp": copy.deepcopy(Pp[215]._p),
     "tbl": copy.deepcopy(Tt[8]._tbl), "lead": copy.deepcopy(Pp[288]._p)}


def txt(i, runs):
    set_p(Pp[i]._p, runs)


def sub(i, old, new):
    t = Pp[i].text
    assert old in t, (i, old)
    txt(i, t.replace(old, new))


# --- Abbreviations: add SAE and BYM2
ab = Tt[0]._tbl
last = copy.deepcopy(ab.findall(qn("w:tr"))[-1])
for tc, val in zip(last.findall(qn("w:tc")), ["SAE", "Small-area estimation", "BYM2", "Besag-York-Mollie spatial model, version 2"]):
    set_cell(tc, val)
ab.append(last)

# --- Executive summary
sub(65, "Bayesian hierarchical beta regression,", "Bayesian hierarchical beta regression, Bayesian small-area estimation with a spatial (BYM2) model,")
t67 = Pp[67].text
i0 = t67.index("The top 20% of reporting LGAs")
i1 = t67.index("Five community archetypes")
t67 = t67[:i0] + (
    f"Two Bayesian methods estimated about 2.1 million zero-dose children in 2026: {m1['national'] / 1e6:.2f} million under Method 1, a "
    "Bayesian hierarchical model with DHIS2-calibrated LGA allocation (773 LGAs), and "
    f"{m2['national'] / 1e6:.2f} million under Method 2, Bayesian small-area estimation (all 774 LGAs). About 150 LGAs hold half "
    f"of the burden and {m2['pareto']['80']} to {m1['pareto']['80']} LGAs are needed to reach 80%; {R['overlap_top155']} LGAs rank in "
    "the top 155 under both methods. ") + t67[i1:]
t67 = t67.replace("(ρ = 0.88)", f"(ρ = {v1n.rho:.2f} for Method 1 and {v2n.rho:.2f} for Method 2)")
t67 = t67.replace("IHME LGA coverage surface (ρ = 0.60)", f"IHME LGA coverage surface (ρ = {v1i.rho:.2f} and {v2i.rho:.2f})")
assert "0.88" not in t67 and "0.60" not in t67
D1 = ROOT.parent / "Domain1_Update_20261009"
d1s = json.loads((D1 / "outputs" / "corrected" / "d1_summary.json").read_text())
d1w = pd.read_excel(D1 / "government_workbook" / "NPHCDA_Domain1_Antigen_EarlyWarning_LGA_2026.xlsx", sheet_name="Worklist (flagged)", header=2)
d1_conf = int(d1w["Decline already visible in 2025"].str.startswith("Yes").sum())
d1n = pd.read_csv(D1 / "outputs" / "corrected" / "d1_national_at_risk_summary.csv").set_index("antigen")
d1a = pd.read_csv(D1 / "outputs" / "corrected" / "D1_additional_antigen_forecast_summary.csv").set_index("Antigen")
d1cv = pd.read_csv(D1 / "figures" / "data" / "D1_F2_cohort_coverage_vs_ndhs.csv").set_index("antigen")
rota25 = d1a.loc[["Rota1", "Rota2", "Rota3"], "2025 observed % of 2024"]
_old = "National forecasts for BCG, Penta1, Penta3, and Measles1 remained above 80%, but 1,645 LGA-antigen forecasts fell below the target, demonstrating that risk is local rather than national."
assert _old in t67
t67 = t67.replace(_old, f"National forecasts for BCG, Penta1, Penta3, and Measles1 remained above 80% of their 2024 level, but {d1s['alerts']:,} of "
                  f"{d1s['forecasts_fitted']:,} LGA-antigen forecasts, in {d1s['lgas_with_any_alert']} of {d1s['lgas_with_any_forecast']} LGAs, fell below "
                  f"that mark, {d1_conf} of them with the decline already visible in 2025, demonstrating that risk is local rather than national.")
_old = "showed stable or improving trajectories with no decline warnings, while IPV2 and the rotavirus series continue to rise."
assert _old in t67
t67 = t67.replace(_old, "showed no decline warnings; IPV2 continues to rise, while rotavirus doses dipped in 2025 and need watching.")
p67_runs = _runs(Pp[67]._p)
txt(67, [(t67, None)])

# --- Table 3 (methods), Domain 5 row
t3 = Tt[3]._tbl.findall(qn("w:tr"))
row5 = [tr for tr in t3 if tr.findall(qn("w:tc"))[0].xpath("string(.)").strip() == "5"][0]
tcs = row5.findall(qn("w:tc"))
set_cell(tcs[1], "Method 1: Bayesian hierarchical Beta regression with DHIS2-calibrated LGA allocation; Method 2: Bayesian "
                 "small-area estimation (survey-anchored, BYM2 spatial model); Getis-Ord Gi*; Ward clustering (k=5)")
set_cell(tcs[2], "State and LGA estimates with credible intervals (773 and 774 LGAs); priority probabilities; hotspots; archetypes")
txt(145, "Bayesian convergence for Domain 5 is satisfactory for both methods: Method 1 maximum R-hat = "
         f"{m1['diagnostics']['max_rhat']:.3f} and Method 2 maximum R-hat = {m2['diagnostics']['max_rhat']:.3f}, with no divergent "
         "transitions. Full step-by-step methods are held in the consortium's technical appendix and the Domain 5 notebooks.")

# --- Section 4.3
txt(148, "4.3 Analytical Coverage of Local Government Areas")
txt(149, "The Domain 5 estimates cover all of Nigeria's 774 LGAs. Method 1 produces estimates for 773 LGAs; Guzamala (Borno) "
         "reported no Penta1 doses to DHIS2 in 2021-2024 (partial reporting began in 2025), and Method 1, which distributes each state's estimate using LGA shares "
         "of reported Penta1 doses, cannot place it. Method 2 estimates all 774 LGAs, including Guzamala, from survey anchors, "
         "local conditions and neighbouring LGAs. Domain 1 forecasts also cover 773 LGAs; Guzamala has no 2024 doses and therefore "
         "no Domain 1 baseline (Section 5.1).")
Pp[150]._p.getparent().remove(Pp[150]._p)
txt(151, "Table 3b. Local government areas covered by the Domain 5 estimates, by zone.")
zrows = []
for z in ZORDER:
    r = ZN[ZN.area == z].iloc[0]
    zrows.append([z, f"{int(r.m1_lgas)} of {int(r.lgas_total)}", f"{int(r.m2_lgas)} of {int(r.lgas_total)}"])
zrows.append(["Nigeria", f"{int(nat.m1_lgas)} of 774", f"{int(nat.m2_lgas)} of 774"])
fill_table(Tt[4]._tbl, zrows, header=["Zone", "LGAs estimated, Method 1", "LGAs estimated, Method 2"])
def set_widths(tbl, widths_in):
    grid = tbl.find(qn("w:tblGrid"))
    for g, w in zip(grid.findall(qn("w:gridCol")), widths_in):
        g.set(qn("w:w"), str(int(w * 1440)))
    for tr in tbl.findall(qn("w:tr")):
        for tc, w in zip(tr.findall(qn("w:tc")), widths_in):
            tcw = tc.find(qn("w:tcPr")).find(qn("w:tcW")) if tc.find(qn("w:tcPr")) is not None else None
            if tcw is not None:
                tcw.set(qn("w:w"), str(int(w * 1440))); tcw.set(qn("w:type"), "dxa")


set_widths(Tt[4]._tbl, [2.2, 2.0, 2.0])
i220 = [i for i, p in enumerate(Pp) if p.text.startswith("Two cautions govern interpretation.")]
assert len(i220) == 1
sub(i220[0], "LGAs without DTP1 reporting, the 44 listed in Table 3b, carry no estimate",
    "LGAs without a DTP1 series in the Domain 3 analysis carry no estimate")
txt(153, "Guzamala (Borno) is the only LGA without a Method 1 estimate. Source: DHIS2 routine antigen export by LGA and month, "
         "2021-2025; GRID3 boundaries (774 LGAs).")
txt(154, "Routine reporting quality still matters. Because Method 1 distributes each state's estimate using LGA shares of reported "
         "Penta1 doses, LGAs whose reported doses are far above or below their estimated child population carry the largest "
         "allocation error; these are also the LGAs where the two methods differ most (Section 5.5.5). The Digital Innovation Hub "
         "should include them in its monthly data-quality checks, beginning with restoring Penta1 reporting in Guzamala.")

# --- Section 5.5 (Method 1)
tri = SF[SF.state.isin(["Sokoto", "Kebbi", "Zamfara"])]
txt(211, "A Bayesian hierarchical Beta regression on the NDHS 2008-2024 series with a DHIS2 Penta1 trend term (Method 1; maximum "
         f"R-hat = {m1['diagnostics']['max_rhat']:.3f}, no divergent transitions) produces state-level posterior forecasts with full "
         "credible intervals for 2026-2028. The three Tier-1 (Critical) states, Sokoto, Kebbi and Zamfara, carry the highest "
         f"predicted rates, each with a 2026 posterior mean near or above {math.floor(tri.rate_2026.min()):.0f}%. The North-West sits "
         "in a band of its own throughout; most other states drift slowly downward, and credible intervals widen after 2024, where "
         "recent survey coverage is thinnest, reflecting honest forecast uncertainty rather than noise.")
top9 = SF.sort_values("children_2026", ascending=False).head(9)
fill_table(Tt[6]._tbl, [[r.state, f"{r.ndhs_2024:.1f}", f"{r.rate_2026:.1f} ({r.rate_2026_lo95:.0f}-{r.rate_2026_hi95:.0f})",
                         k(r.children_2026)] for r in top9.itertuples()])
txt(213, [(f"Burden is the posterior-mean rate applied to each state's 12-to-23-month birth cohort (state total "
           f"{SF.children_2026.sum() / 1e6:.2f} million). LGA totals in Section 5.5.2 sum to {m1['national'] / 1e6:.2f} million because "
           "LGA rates are bounded between 1% and 99% during allocation.", None)])
nw, ne, nc = (ZN[ZN.area == z].iloc[0] for z in ("North West", "North East", "North Central"))
t214 = Pp[214].text.replace("about 1.08 million", f"about {nw.m1_children / 1e6:.2f} million").replace(
    "roughly 330,000 each", f"roughly {round((ne.m1_children + nc.m1_children) / 2, -4):,.0f} each")
r214 = _runs(Pp[214]._p)
txt(214, t214)
replace_image(doc, Pp[215]._p, DFIG / "R_09_state_trajectories.png", 6.27)
txt(216, "Figure 9. State-level zero-dose: NDHS 2008-2024 observations and Method 1 forecasts for 2026-2028 (shaded bands are 95% "
         "credible intervals).")
replace_image(doc, Pp[217]._p, DFIG / "R_10_state_2026_intervals.png", 5.2)
txt(218, "Figure 10. Predicted state zero-dose rate (2026) with 95% credible intervals, by zone, against NDHS 2024 observed values.")
replace_image(doc, Pp[219]._p, FIG / "M1_01_lga_rate_and_hotspots.png", 6.27)
txt(220, "Figures 11-12. Method 1 LGA zero-dose rate (2026) and Getis-Ord Gi* hotspot clusters, 773 LGAs.")
replace_image(doc, Pp[221]._p, FIG / "M1_02_state_hotspots_2026_2028.png", 6.27)
txt(222, "Figure 13. Forecast state zero-dose hotspots, 2026-2028 (Getis-Ord Gi*, Queen contiguity; Method 1).")
txt(223, f"A ranked list of all {m1['lgas']} modelled LGAs orders the burden for microplanning. The burden is concentrated, but less "
         f"sharply than an 80/20 distribution would imply. The highest-burden 155 LGAs, the top 20% of LGAs, contain "
         f"{m1['top155_share']:.0f}% of estimated zero-dose children. Half of the burden is reached with {m1['pareto']['50']} LGAs, "
         f"60% with {m1['pareto']['60']} and 80% with {m1['pareto']['80']}, about {P[80].m1_pct_of_lgas:.0f}% of those modelled.")
txt(224, f"The practical consequence is a two-wave effort. A first wave targeting about {m1['pareto']['50']} LGAs would reach half of "
         "unreached children and is fundable as a single package. Coverage of 80% cannot be achieved from that first wave alone, so "
         f"the second, extending toward {m1['pareto']['80']} LGAs, should be planned alongside it rather than treated as contingent.")
replace_image(doc, Pp[225]._p, FIG / "M1_03_pareto.png", 5.6)
txt(226, f"Figure 14. Pareto concentration of zero-dose burden across {m1['lgas']} LGAs (Method 1; total "
         f"{m1['national'] / 1e6:.2f} million children), with the LGAs needed to reach 50%, 60% and 80%.")

# 5.5.1 archetypes
sub(230, "approximately 1.50 million of the 2.09 million modelled zero-dose children, about 72% of the national burden,",
    f"approximately {a12_m1 / 1e6:.2f} million of the {m1['national'] / 1e6:.2f} million modelled zero-dose children (Method 1), "
    f"about {a12_s1:.0f}% of the national burden ({a12_s2:.0f}% under Method 2),")
t7 = Tt[7]._tbl.findall(qn("w:tr"))[1:]
for a_id, tr in zip(range(1, 6), t7):
    tcs = tr.findall(qn("w:tc"))
    set_cell(tcs[3], f"{k(A[a_id].m1_children)}\n({A[a_id].m1_share:.0f}%)")
    set_cell(tcs[4], f"{A[a_id].m1_mean_rate:.1f}%")
sub(233, "Percentages are of the 2.09 million national total.",
    f"Zero-dose children and mean LGA rates are Method 1 estimates; percentages are of the {m1['national'] / 1e6:.2f} million "
    "national total. Method 2 values are given in Figure 15 and the government workbook.")
replace_image(doc, Pp[234]._p, FIG / "A_01_archetype_map_burden.png", 6.27)
txt(235, "Figure 15. LGA archetypes (774 LGAs) and each archetype's share of children and of modelled zero-dose children under both methods.")

# 5.5.2 ranked list
txt(238, "Every modelled LGA is ranked worst-to-best by estimated zero-dose burden, providing the prioritisation ordering required "
         f"for microplanning. As Figure 14 shows, the {m1['pareto']['50']} highest-burden LGAs reach half of all zero-dose children, "
         f"{m1['pareto']['60']} reach 60% and {m1['pareto']['80']} reach 80%. The ten highest-burden LGAs under Method 1 are shown "
         "below; the full ranked list for both methods, filterable by state and priority tier, is held in the platform and in the "
         "government workbook.")
top10 = L[L.m1_rank.notna()].sort_values("m1_rank").head(10)
fill_table(Tt[8]._tbl, [[int(r.m1_rank), r.state, r.lga_clean, r.zone, k(r.m1_children), f"{r.m1_rate:.1f}"] for r in top10.itertuples()])
txt(239, "Table 6b. The ten highest-burden LGAs by modelled zero-dose children, 2026 (Method 1).")
sub(241, "Source: Domain 5 zero-dose model, 2026;", f"Source: {M1N}, 2026;")
sc = top10.state.value_counts()
txt(242, [("Two features of this ranking have direct programme implications. First, the top of the list is concentrated in "
           + " and ".join(sc.index[:2]) + f", which hold {int(sc.iloc[:2].sum())} of the ten highest-burden LGAs, while the large "
           "birth cohorts of Kano, Katsina and Kaduna place many of their LGAs further down the list; rate-based and burden-based "
           f"prioritisation therefore produce different lists, and microplanning needs both. Second, {m1['capped']} LGAs sit at the "
           "99% upper bound of the Method 1 allocation, which should trigger verification of the local denominator and of reported "
           "doses before campaign resources are committed.", None)])

# 5.5.3 maternal care
anc_n = 766
txt(245, "The LGAs with the weakest antenatal and delivery care are the same LGAs missing the most zero-dose children. Across the "
         f"{anc_n} LGAs with maternal-care data and a Method 1 estimate, antenatal care with four or more visits correlates with the "
         f"modelled zero-dose rate at r = {CAPS['_anc_r_m1']:.2f}, and facility delivery at r = {CAPS['_deliv_r_m1']:.2f}. Both "
         "associations are strong and consistent, and they indicate an operational entry point that immunization delivery does not "
         "currently use.")
sub(246, "approximately 1.50 million of the roughly 2.09 million", f"approximately {a12_m1 / 1e6:.2f} million of the {m1['national'] / 1e6:.2f} million")
replace_image(doc, Pp[247]._p, FIG / "A_02_maternal_care.png", 6.27)
txt(248, f"Figure 16. Antenatal care (four or more visits) against the Method 1 zero-dose rate ({anc_n} LGAs), and maternal-care "
         "indicators and zero-dose under both methods, by archetype.")
t9 = Tt[9]._tbl.findall(qn("w:tr"))[1:]
for a_id, tr in zip(range(1, 6), t9):
    tcs = tr.findall(qn("w:tc"))
    set_cell(tcs[4], f"{A[a_id].m1_mean_rate:.1f}")
    set_cell(tcs[5], k(A[a_id].m1_children))
sub(252, "modelled zero-dose)", "Method 1 modelled zero-dose)")

# --- New 5.5.4 and 5.5.5 after the RMNCH-EPI note (P252)
c = Cursor(doc, Pp[252]._p, T)
c.para("h2", "5.5.4 Method 2: Bayesian Small-Area Estimation for All 774 LGAs")
c.para("body", "Method 1 estimates zero-dose at state level and distributes each state's estimate to its LGAs using routine DHIS2 "
               "Penta1 reporting. A second, independent method was built to estimate LGA zero-dose directly, without routine coverage "
               "data, so that LGA priorities could be checked against an approach that does not depend on reported doses or "
               "administrative denominators.")
c.para("body", "Method 2 is a Bayesian small-area estimation (SAE) model. Each LGA has its own zero-dose rate on the logit scale. "
               "The population-weighted average of a state's LGA rates is linked to that state's NDHS zero-dose estimate (2008, 2013, "
               "2018 and 2023-24) through a measurement-error likelihood that allows for survey sampling error (design effect 2), so "
               "state totals stay consistent with the surveys. Differences between LGAs within a state are explained by six LGA "
               "covariates (a maternal-care index combining antenatal care with four or more visits and facility delivery, improved "
               "water, the Meta Relative Wealth Index, travel time to a health facility, conflict events and poverty), by a BYM2 "
               "spatial random effect on the GRID3 LGA adjacency graph (2,165 neighbour pairs), and by zone effects. The 2026 "
               "estimate carries forward each state's survey-anchored level with added process uncertainty. The model was fitted "
               "in PyMC with four chains of 8,000 draws after 3,000 tuning steps; every LGA receives a posterior mean, a 95% "
               "credible interval, and a probability of being among the 155 highest-burden LGAs.")
c.figure(FIG / "M2_01_model_schematic.png", "Figure 16b. Structure of the Bayesian small-area model (Method 2).")
nr2 = nat.m2_rate
c.para("body", f"Method 2 estimates {m2['national'] / 1e6:.2f} million zero-dose children in 2026 (95% credible interval "
               f"{m2['national_lo95'] / 1e6:.2f} to {m2['national_hi95'] / 1e6:.2f} million), a national rate of {nr2:.1f}%. The North-West "
               f"holds {ZN[ZN.area == 'North West'].iloc[0].m2_share:.0f}% of the total at a zone rate of "
               f"{ZN[ZN.area == 'North West'].iloc[0].m2_rate:.1f}%. The 155 highest-burden LGAs hold {m2['top155_share']:.0f}% of zero-dose "
               f"children, and {m2['near_certain_top155']} LGAs are among the top 155 in at least 90% of posterior simulations. Credible "
               "intervals are widest where survey anchors and covariates leave most room for error, which identifies the LGAs where "
               "local verification is most valuable before large investments.")
c.figure(FIG / "M2_02_rate_and_uncertainty_maps.png", "Figure 16c. Method 2 LGA zero-dose rate for 2026 and the width of its 95% credible interval, 774 LGAs.")
c.figure(FIG / "M2_03_burden_and_priority_maps.png", "Figure 16d. Method 2 modelled zero-dose children per LGA and the probability that each LGA is among the 155 highest-burden LGAs.")
t20 = L.nsmallest(10, "m2_rank")
c.table("Table 6d. The ten highest-burden LGAs under Method 2, 2026.",
        [["Rank", "State", "LGA", "Zero-dose children (95% CrI)", "Rate % (95% CrI)", "P(top 155)"]] +
        [[int(r.m2_rank), r.state, r.lga_clean, f"{k(r.m2_children)} ({k(r.m2_children_lo95)}-{k(r.m2_children_hi95)})",
          f"{r.m2_rate:.1f} ({r.m2_rate_lo95:.0f}-{r.m2_rate_hi95:.0f})", f"{r.m2_p_top155:.2f}"] for r in t20.itertuples()],
        [0.5, 0.85, 1.25, 1.65, 1.25, 0.77],
        f"Model estimate. Source: {M2N}, 2026. CrI = credible interval; P(top 155) = share of 4,000 posterior draws in which the LGA "
        "ranks among the 155 highest-burden LGAs.")
cov0 = COV.iloc[0]
c.para("body", "Maternal care is the strongest and most certain predictor of differences between LGAs within a state: one standard "
               f"deviation higher on the maternal-care index is associated with a log-odds change of {cov0['mean']:.2f} (95% credible "
               f"interval {cov0.lo95:.2f} to {cov0.hi95:.2f}), an odds ratio of {math.exp(cov0['mean']):.2f}. Improved water is also "
               "associated with lower zero-dose; the remaining effects are smaller and their intervals include zero. These are "
               "partial associations used to distribute survey levels within states, not causal effects.")
c.figure(FIG / "M2_06_covariate_effects.png", "Figure 16e. Posterior covariate effects in the Bayesian small-area model (log-odds of zero-dose per standard deviation; 80% and 95% credible intervals).", 5.6)
d2 = m2["diagnostics"]
c.para("body", f"The model converged cleanly: maximum R-hat {d2['max_rhat']:.3f}, minimum bulk effective sample size "
               f"{k(d2['min_ess_bulk'])}, minimum tail effective sample size {k(d2['min_ess_tail'])} and no divergent transitions across "
               f"{k(d2['parameters'])} monitored parameters. Posterior predictive intervals contain "
               f"{m2['survey_ppc_coverage95'] * 100:.0f}% of the 148 NDHS state-round observations, and the 2026 state estimates rank the "
               f"37 states in close agreement with the NmDHS 2025-26, which was not used in fitting (ρ = {v2n.rho:.2f}; mean absolute "
               f"error {v2n.MAE_pp:.1f} percentage points; bias {v2n.bias_pp:+.1f}).")
c.figure(FIG / "M2_05_diagnostics.png", "Figure 16f. Method 2 model performance: fit to all NDHS rounds, independent NmDHS 2025-26 check, and Markov chain Monte Carlo diagnostics.")
c.para("body", "Method 2 does not use routine administrative coverage. Dividing DHIS2 Penta1 doses by the estimated LGA cohort gives "
               f"coverage above 100% in {CAPS['_admin_gt100'] * 100:.0f}% of LGAs, and state zero-dose derived from routine data alone "
               f"agrees poorly with the NmDHS 2025-26 (ρ = {CAPS['_admin_rho']:.2f}; mean absolute difference "
               f"{CAPS['_admin_mae']:.0f} percentage points). Routine data remain essential for tracking trends and data quality, but "
               "survey-anchored estimates are more reliable for the level of zero-dose in each LGA.")
c.figure(FIG / "M2_07_admin_vs_survey.png", "Figure 16g. Routine administrative Penta1 coverage compared with survey-anchored small-area estimates.")

c.para("h2", "5.5.5 Comparison of Method 1 and Method 2")
c.para("body", f"The two methods agree closely on national, zone and state totals. Nationally they differ by {nat.difference_pct:.1f}% "
               f"({m1['national'] / 1e6:.2f} and {m2['national'] / 1e6:.2f} million), the 95% intervals overlap in every zone, and both place "
               f"about half of all zero-dose children in the North-West. Across the 37 states, the rank correlation of modelled zero-dose "
               f"children between methods is ρ = {rs_state:.2f}.")
zt = [["Zone", "LGAs (M1/M2)", "Method 1 zero-dose children (95% interval)", "M1 rate %", "Method 2 zero-dose children (95% CrI)",
       "M2 rate %", "M2 vs M1", "NmDHS 2025-26 %"]]
for z in ZORDER + ["National"]:
    r = ZN[ZN.area == z].iloc[0]
    nm = ZV[ZV.area == z].nmdhs_2025_26
    zt.append([("Nigeria" if z == "National" else z), f"{int(r.m1_lgas)}/{int(r.m2_lgas)}",
               f"{k(r.m1_children)} ({k(r.m1_lo95)}-{k(r.m1_hi95)})", f"{r.m1_rate:.1f}",
               f"{k(r.m2_children)} ({k(r.m2_lo95)}-{k(r.m2_hi95)})", f"{r.m2_rate:.1f}", f"{r.difference_pct:+.1f}%",
               (f"{nm.iloc[0]:.1f}" if len(nm) else "-")])
c.table("Table 6e. Modelled zero-dose children in 2026 by zone, Method 1 and Method 2.", zt,
        [0.95, 0.6, 1.35, 0.5, 1.35, 0.5, 0.5, 0.52],
        f"Model estimate, 2026. {M1N}; {M2N}. NmDHS 2025-26 zone rates for reference (not used in fitting).")
c.figure(FIG / "C_01_zone_comparison.png", "Figure 16h. Modelled zero-dose children by zone, 2026, under both methods, with 95% intervals.", 5.6)
so = ST.assign(zo=ST.zone.map({z: i for i, z in enumerate(ZORDER)})).sort_values(["zo", "m1_children"], ascending=[True, False])
c.table("Table 6f. Modelled zero-dose children in 2026 by state, Method 1 and Method 2.",
        [["State", "Zone", "Method 1 children", "M1 rate %", "Method 2 children", "M2 rate %", "NmDHS 2025-26 %"]] +
        [[r.state, r.zone, k(r.m1_children), f"{r.m1_rate:.1f}", k(r.m2_children), f"{r.m2_rate:.1f}", f"{r.nmdhs_2025_26:.1f}"]
         for r in so.itertuples()],
        [1.0, 1.0, 1.0, 0.7, 1.0, 0.7, 0.87],
        f"Model estimate, 2026. {M1N}; {M2N}. States ordered by zone and Method 1 burden.")
c.para("body", f"At LGA level the methods agree on which LGAs carry the burden (rank correlation ρ = {R['rank_corr']:.2f} across 773 "
               f"LGAs; {R['overlap_top155']} of the top 155 LGAs are shared), but individual LGAs can differ substantially. Method 1 "
               "follows each LGA's share of reported Penta1 doses: very high reported doses push its rate toward the 1% floor (for "
               "example Rabah and Goronyo in Sokoto) and low reported doses push it up (for example Sokoto North). Method 2 is not "
               "affected by reporting volumes. LGAs with large differences between the methods are therefore priorities for "
               "data-quality review as well as for programme attention.")
c.figure(FIG / "C_05_lga_scatter.png", "Figure 16i. LGA-level agreement between Method 1 and Method 2, 773 LGAs.")
pt = [["Share of zero-dose children", "Method 1: LGAs (% of LGAs)", "Method 2: LGAs (% of LGAs)"]]
for sh in (50, 60, 80):
    r = P[sh]
    pt.append([f"{sh}%", f"{int(r.m1_lgas)} ({r.m1_pct_of_lgas:.1f}%)", f"{int(r.m2_lgas)} ({r.m2_pct_of_lgas:.1f}%)"])
pt.append(["Share held by the top 155 LGAs (20%)", f"{T155.m1_share:.1f}% ({T155.m1_lo95:.1f}-{T155.m1_hi95:.1f})",
           f"{T155.m2_share:.1f}% ({T155.m2_lo95:.1f}-{T155.m2_hi95:.1f})"])
c.table("Table 6g. LGAs needed to reach 50%, 60% and 80% of zero-dose children, by method.", pt, [2.3, 2.0, 1.97],
        "Model estimate, 2026. LGAs ranked from highest to lowest modelled burden; intervals are 95% credible intervals.")
c.figure(FIG / "C_04_pareto_both_methods.png", "Figure 16j. Cumulative share of modelled zero-dose children by number of LGAs targeted, both methods.", 5.4)
c.table("Table 6h. Which estimate to use for which decision.",
        [["Decision", "Recommended estimate", "Reason"],
         ["National, zone and state totals", "Either; report Method 1 with Method 2 as the sensitivity range",
          f"Totals agree within {abs(nat.difference_pct):.1f}% nationally; state ranks agree (ρ = {rs_state:.2f}); both match NmDHS 2025-26"],
         ["Ranking LGAs within a state", "Method 2",
          f"Covers all 774 LGAs with full uncertainty; within-state agreement with IHME ρ = {v2w.rho:.2f} vs {v1w.rho:.2f}"],
         ["Selecting priority LGAs for funding", f"The {R['overlap_top155']} LGAs in the top 155 under both methods first",
          "Robust to the choice of method; then add LGAs with high Method 2 priority probability"],
         ["Tracking change and early warning", "Method 1", "Uses the DHIS2 trend and updates with monthly routine data"],
         ["Data-quality review", "LGAs where the methods differ most", "Large gaps flag reported doses out of line with population"]],
        [1.8, 2.1, 2.37])

# --- 5.9 validation
txt(288, [("State level, against a survey that did not exist. ", True),
          ("The Nigeria mini Demographic and Health Survey (NmDHS) 2025-26 was released after this modelling concluded, which makes "
           "it a genuine out-of-sample test rather than a retrospective fit. Method 1 and the new survey rank the 37 states in almost "
           f"the same order (Spearman ρ = {v1n.rho:.2f}); Method 2 gives ρ = {v2n.rho:.2f}. On levels, the North-West forecast of "
           f"{ZV[ZV.area == 'North West'].iloc[0].m1_rate:.0f}% sits within one point of the survey's "
           f"{ZV[ZV.area == 'North West'].iloc[0].nmdhs_2025_26:.0f}%. Ordering and magnitude both hold against data the models never saw.", False)])
replace_image(doc, Pp[289]._p, FIG / "M1_04_nmdhs_validation.png", 6.27)
txt(290, f"Figure 25. Prospective validation: the 2026 Method 1 state forecasts against the NmDHS 2025-26 survey, 37 states (ρ = {v1n.rho:.2f}).")
txt(291, [("LGA level, against an independently constructed surface. ", True),
          ("The LGA ranking was compared with the IHME Local Burden of Disease DTP1 admin-2 surface, built by a different team using "
           f"different methods and inputs. Across {int(v1i.n)} matched LGAs the Method 1 rank correlation is ρ = {v1i.rho:.2f}, and "
           f"most LGAs fall in the same or an adjacent priority third; only {CAPS['_m1_ihme_opposite']} are placed in opposite thirds "
           f"and are flagged for review. Method 2 agrees more closely (ρ = {v2i.rho:.2f}), including within states (ρ = {v2w.rho:.2f} "
           f"against {v1w.rho:.2f} for Method 1). Absolute levels differ, as expected: the IHME surface ends in 2018 and these "
           "estimates are for 2026, and eight years shift the levels even where the geography agrees.", False)])
replace_image(doc, Pp[292]._p, FIG / "M1_05_ihme_concordance.png", 6.27)
txt(293, f"Figure 26. Convergent validity: the 2026 Method 1 LGA estimates against the IHME Local Burden of Disease DTP1 surface (ρ = {v1i.rho:.2f}).")


def ci(r):
    return f"{r.rho:.2f} ({r.ci_lo:.2f} to {r.ci_hi:.2f})"


c = Cursor(doc, Pp[293]._p, T)
c.table("Table 7b. National comparison and agreement with independent data, both Domain 5 methods.",
        [["Measure", "Method 1", "Method 2 (SAE)"],
         ["LGAs estimated", str(m1["lgas"]), str(m2["lgas"])],
         ["Zero-dose children, 2026 (95% interval)", f"{m1['national'] / 1e6:.2f} million ({m1['national_lo95'] / 1e6:.2f}-{m1['national_hi95'] / 1e6:.2f})",
          f"{m2['national'] / 1e6:.2f} million ({m2['national_lo95'] / 1e6:.2f}-{m2['national_hi95'] / 1e6:.2f})"],
         ["National zero-dose rate", f"{nat.m1_rate:.1f}%", f"{nat.m2_rate:.1f}%"],
         ["NmDHS 2025-26, 37 states: Spearman ρ (95% CI)", ci(v1n), ci(v2n)],
         ["NmDHS 2025-26: mean absolute error / bias (points)", f"{v1n.MAE_pp:.1f} / {v1n.bias_pp:+.1f}", f"{v2n.MAE_pp:.1f} / {v2n.bias_pp:+.1f}"],
         [f"IHME DTP1 2018, {int(v1i.n)} LGAs: Spearman ρ (95% CI)", ci(v1i), ci(v2i)],
         ["IHME DTP1 2018, within states: Spearman ρ (95% CI)", ci(v1w), ci(v2w)],
         ["States with positive within-state agreement", f"{int(v1w.states_positive)} of {int(v1w.states_tested)}",
          f"{int(v2w.states_positive)} of {int(v2w.states_tested)}"],
         ["LGAs in the top 155 under both methods", str(R["overlap_top155"]), str(R["overlap_top155"])]],
        [2.6, 1.85, 1.82],
        "ρ = Spearman rank correlation; 95% CI from 2,000 bootstrap resamples. All p < 0.001 except Method 1 within states "
        f"(p = {v1w.p_value:.2f}). Within-state agreement uses ranks computed within each state and pooled.")
c.figure(FIG / "C_03_agreement_independent_data.png", "Figure 26b. Spearman rank correlation of each Domain 5 method with independent data: NmDHS 2025-26 (states) and IHME 2018 (LGAs, overall and within state).")

# --- Conclusions, recommendations, limitations
txt(434, [("Target the convergent Tier-1 North-West first. ", True),
          ("Concentrate the largest, most sustained package on Sokoto, Kebbi, and Zamfara, with Niger, Katsina, Kaduna, and Jigawa "
           "close behind, using the ranked LGA list for microplanning rather than whole-state campaigns. Sequence the investment in "
           f"two waves: an initial wave across about {m1['pareto']['50']} LGAs reaches half of missed children and is fundable as a "
           f"single package, with expansion to about {m1['pareto']['80']} LGAs required to reach 80%. The {R['overlap_top155']} LGAs "
           "ranked in the top 155 by both Domain 5 methods should form the core of the first wave. Budget planning should provide "
           "for the second wave from the outset rather than treating it as contingent.", False)])
txt(452, "Method 1 does not estimate Guzamala (Borno), which reported no Penta1 doses in 2021-2024, and its LGA allocation follows "
         "reported Penta1 doses, so LGAs whose reported doses are out of line with their population carry larger error. Method 2, "
         "which does not use routine coverage, provides the cross-check (Section 5.5.5); it relies on modelled covariate surfaces "
         "(2016-2024) and survey anchors, and its LGA estimates are model-based. Priority LGAs should be confirmed locally before "
         "large investments.")

# --- Section 5.1 Domain 1 (corrected DHIS2 counts, 773 LGAs)
n = d1n
txt(157, "Domain 1 is read two ways. An early-warning view compares each antigen with its own 2024 baseline, treating the 80% line as a decline "
         "tripwire. At the national level all four tracer antigens (BCG, Penta1, Penta3, Measles1) are forecast to stay above their 2024 level "
         f"throughout the next 6-12 months; the lowest national points are {n.loc['BCG', 'min_forecast_pct']:.0f}% for BCG "
         f"({n.loc['BCG', 'min_when']}), {n.loc['Penta1', 'min_forecast_pct']:.0f}% for Penta1, {n.loc['Penta3', 'min_forecast_pct']:.0f}% for "
         f"Penta3 and {n.loc['Measles1', 'min_forecast_pct']:.0f}% for Measles1. The national aggregate conceals the operational risk. Each of the "
         f"{d1s['lgas_with_any_forecast']} LGAs with 2024 data carries its own forecast for each of the four tracer antigens, giving "
         f"{d1s['forecasts_fitted']:,} LGA-and-antigen forecasts. Of these, {d1s['alerts']:,} dip below the 80% mark within the next 6 to 12 months, "
         f"across {d1s['lgas_with_any_alert']} LGAs. In {d1_conf} of them the decline is already visible in 2025 data (2025 doses below 80% of 2024), "
         f"and these should be followed up first; the remaining {d1s['alerts'] - d1_conf:,} are projected dips in series still close to their 2024 "
         "level. The risk is real but local, and must be managed LGA by LGA rather than read off the national average.")
replace_image(doc, Pp[158]._p, D1 / "figures" / "D1_F1_national_tracer_forecasts.png", 6.27)
txt(159, "Figure 1a. National tracer-antigen doses as a percent of their 2024 level, with Prophet forecasts and 95% prediction intervals.")
replace_image(doc, Pp[160]._p, D1 / "figures" / "D1_F2_cohort_coverage_vs_ndhs.png", 5.4)
txt(161, "Figure 1b. 2026 administrative coverage of the eligible cohort against NDHS 2024 survey coverage.")
cv = d1cv
c1 = Cursor(doc, Pp[161]._p, T)
c1.para("body", "Administrative coverage of the eligible cohort tells a different story from the survey. Dividing forecast 2026 doses by the "
                f"12-23-month cohort gives {cv.admin_coverage_2026_pct.min():.0f}-{cv.admin_coverage_2026_pct.max():.0f}% for the four tracer "
                f"antigens, against {cv.ndhs_2024_pct.min():.0f}-{cv.ndhs_2024_pct.max():.0f}% in NDHS 2024. Routine doses exceed the projected "
                "cohort, which points to denominators that are too small and to doses counted where children are vaccinated rather than where "
                "they live. This is why the early-warning flag uses each area's own 2024 level, and why Domain 5 anchors local estimates to "
                "the surveys.")
c1.figure(D1 / "figures" / "D1_F4_lga_flag_map.png", "Figure 1c. Number of the four tracer antigens with an early-warning flag, by LGA.", 5.4)
c1.figure(D1 / "figures" / "D1_F3_lga_early_warning_composite.png", "Figure 1d. LGA early-warning flags by antigen: severity, states with the most "
          "flagged LGAs, and illustrative LGAs.", 6.27)
c1.para("note", f"Model estimate. Forecasts below 0% ({d1s['forecasts_below_zero']} of {d1s['forecasts_fitted']:,}) indicate an erratic LGA series, "
                "for example a sudden change in reporting, and should be verified in DHIS2 before action. The full LGA list is in the Domain 1 "
                "government workbook.")
_p165 = Pp[165].text
_o = "Four recently introduced series, IPV2 and Rotavirus doses 1 to 3, are still climbing as the introductions bed in;"
assert _o in _p165
txt(165, _p165.replace(_o, "Four recently introduced series, IPV2 and Rotavirus doses 1 to 3, follow different paths: IPV2 is still climbing "
                       f"(2025 doses {d1a.loc['IPV2', '2025 observed % of 2024']:.0f}% of 2024), while rotavirus doses fell to "
                       f"{rota25.min():.0f}-{rota25.max():.0f}% of their 2024 level in 2025 and should be watched;"))
replace_image(doc, Pp[167]._p, D1 / "figures" / "D1_F5_additional_antigens.png", 6.27)
txt(168, "Figure 1e. Forecast trajectories for the nine additional antigens against the 80% at-risk-of-decline mark (Prophet, DHIS2 2021-2025, "
         "projected to 2028).")
replace_image(doc, Pp[177]._p, D1 / "figures" / "D1_F6_opv3_worked_example.png", 5.6)
txt(178, "Figure 1f. Worked example: the dotted line is 80% of OPV3's own 2024 doses. The forecast remains above it, so OPV3 is not flagged.")
_p455 = Pp[455].text
_o = "Denominator instability (coverage exceeding 100% in some states)"
assert _o in _p455
txt(455, _p455.replace(_o, "Denominator instability (administrative coverage exceeding 100% nationally for all four tracer antigens and in most LGAs)"))

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT))
print("saved", OUT)
