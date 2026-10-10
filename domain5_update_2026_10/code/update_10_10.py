"""Build the 10.10.2026 versions of the NPHCDA consolidated report and RI team deck.

Starts from the 10.9.2026 report and deck (never modified) and adds the validation, calibration and
uncertainty-aware prioritization results for Method 2 and the contextual-profile and candidate-package
results, using the journal-quality figures and tables in ScienceLab/. Archetype labels are replaced by
descriptive labels that name only what the input indicators measure.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
RD = ROOT / "report_and_deck"
SRC_DOC = RD / "Final_NPHCDA_Consolidated_Report_Domain5_Two_Methods_10.9.2026.docx"
OUT_DOC = RD / "Final_NPHCDA_Consolidated_Report_Domain5_Two_Methods_10.10.2026.docx"
SRC_PPT = BASE.parent / "nphcda-zerodose-git" / "domain5_update_2026_10" / "report_and_deck" / \
    "NPHCDA_ZeroDose_RI_Team_Presentation_Domain5_Two_Methods_10.9.2026.pptx"
OUT_PPT = RD / "NPHCDA_ZeroDose_RI_Team_Presentation_Domain5_Two_Methods_10.10.2026.pptx"
FIG = ROOT / "figures"
SL = BASE / "ScienceLab"
FA, FB = SL / "Study_A" / "Figures", SL / "Study_B" / "Figures"
TA = pd.read_excel(SL / "Study_A" / "Tables" / "Study_A_Tables.xlsx", sheet_name=None)
TB = pd.read_excel(SL / "Study_B" / "Tables" / "Study_B_Tables.xlsx", sheet_name=None)
BMAT = pd.read_csv(BASE / "WorldBank_Conference_2026_Refinement" / "02_Study_B" / "tables" / "B_R4_barrier_intervention_matrix.csv")

M1N = "Method 1: Bayesian hierarchical model with DHIS2-calibrated LGA allocation"
M2N = "Method 2: Bayesian small-area estimation (SAE)"
PLAB = {1: "Deprived northern rural", 2: "Low maternal care, high conflict exposure", 3: "Geographically isolated",
        4: "Near-average", 5: "Relatively advantaged"}


def a_fig(n):
    return next(FA.glob(f"Figure_A{n}_*.png"))


def b_fig(n):
    return next(FB.glob(f"Figure_B{n}_*.png"))


# ================================================================== REPORT
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
    for i, part in enumerate(text.split("\n")):
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
        p_el.append(_mk_run(tmpl if tmpl is not None else first, text, b))


def set_cell(tc, text, bold=None):
    ps = tc.findall(qn("w:p"))
    for p in ps[1:]:
        tc.remove(p)
    rs = _runs(ps[0])
    tmpl = rs[0] if rs else ps[0].makeelement(qn("w:r"), {})
    set_p(ps[0], [(text, bold)], tmpl_run=tmpl)


def make_table(template_tbl, rows, widths_in):
    t = copy.deepcopy(template_tbl)
    grid = t.find(qn("w:tblGrid"))
    for g in list(grid):
        grid.remove(g)
    tblw = t.find(qn("w:tblPr")).find(qn("w:tblW"))
    if tblw is not None:
        tblw.set(qn("w:w"), str(int(sum(widths_in) * 1440))); tblw.set(qn("w:type"), "dxa")
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
        tr = copy.deepcopy(htr if i == 0 else btr)
        tcs = tr.findall(qn("w:tc"))
        tc0, tc1 = copy.deepcopy(tcs[0]), copy.deepcopy(tcs[1] if len(tcs) > 1 else tcs[0])
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


def cell_sub(tbl, r, c, new):
    set_cell(tbl.rows[r].cells[c]._tc, new)


def build_report():
    doc = Document(str(SRC_DOC))
    P = list(doc.paragraphs)
    Tt = list(doc.tables)
    assert P[251].text.startswith("5.5.1") and P[298].text.startswith("5.5.5") and P[314].text.startswith("5.6 Domain 6")
    T = {"h2": copy.deepcopy(P[298]._p), "body": copy.deepcopy(P[299]._p), "tabcap": copy.deepcopy(P[300]._p),
         "note": copy.deepcopy(P[301]._p), "imgp": copy.deepcopy(P[302]._p), "figcap": copy.deepcopy(P[303]._p),
         "tbl": copy.deepcopy(P[300]._p.getnext())}

    # ---- label and claim corrections in existing Domain 5 text
    set_p(P[99]._p, P[99].text.replace("differentiate interventions by community archetype",
                                       "differentiate interventions by each LGA's measured barriers"))
    set_p(P[253]._p, "The clustering uses agglomerative (Ward) linkage on fifteen indicators in six domains (socioeconomic, maternal "
                     "care, nutrition, access, insecurity and programme performance) drawn from IHME, DHS, the Meta Relative Wealth "
                     "Index, Weiss travel-time surfaces and ACLED conflict data, across all 774 LGAs. Five archetypes, also called "
                     "contextual profiles, are retained, each with a distinct barrier signature. Their labels describe only what the "
                     "indicators measure; Section 5.5.7 reports how stable the profiles are and how candidate packages are assigned.")
    set_p(P[254]._p, "The clustering also provides a consistency check on the zero-dose surface. The archetypes were built from "
                     "contextual indicators: education, nutrition, poverty, wealth, travel time, water, maternal care, conflict and "
                     "the IHME 2016 DTP1-3 dropout surface. No zero-dose or coverage estimate entered the clustering. The archetypes "
                     "nevertheless reproduce the geography of missed children: Archetypes 1 and 2 together hold approximately 1.52 "
                     "million of the 2.10 million modelled zero-dose children (Method 1), about 72% of the national burden (77% under "
                     "Method 2), across 328 of the 774 LGAs. Because several indicators are also inputs to the estimation models, this "
                     "is a consistency check rather than independent validation; the independent state-level check against the "
                     "NmDHS 2025-26 is reported in Section 5.5.7.")
    set_p(P[257]._p, P[257].text.rstrip() + " Labels describe measured indicators only; population mobility, migration, riverine "
                                            "settlement and informal settlements are not measured by any input.")
    set_p(P[259]._p, "Figure 15. Contextual profiles (community archetypes) of the 774 LGAs and each profile's share of children "
                     "and of modelled zero-dose children under both methods.")
    replace_image(doc, P[258]._p, FIG / "P_01_profile_map_burden.png", 6.27)
    set_p(P[270]._p, P[270].text.replace("Remote Rural / Hard-to-Reach and Conflict-Affected / Nomadic",
                                         "deprived northern rural, and low maternal care with high conflict exposure"))
    assert "deprived northern rural" in P[270].text
    lead = "Differentiate delivery by measured barriers."
    set_p(P[501]._p, [(lead + " ", True),
                      ("Start from a state package and refine it with each LGA's flagged barriers (Table 6l), using the "
                       "contextual profiles as a planning summary; one national package agrees least with LGA barrier profiles. "
                       "Validate components locally through microplanning, and treat demand and supply as separate maps.", False)])
    set_p(P[516]._p, P[516].text.replace("archetype-matched packages in Tables 6 and 6c",
                                         "archetype-matched packages in Tables 6 and 6c and the LGA barrier-to-intervention "
                                         "mapping in Table 6l"))
    assert "Table 6l" in P[516].text

    # Table 6 (tables[7]) and Table 6c (tables[9])
    t6, t6c = Tt[7], Tt[9]
    ex = {1: "Kano, Jigawa, Katsina", 2: "Borno, Niger, Adamawa", 3: "Cross River, Bayelsa, Delta", 4: "Osun, Akwa Ibom, Oyo",
          5: "Imo, Anambra, Lagos"}
    cell_sub(t6, 0, 1, "Profile (example states)")
    for k in range(1, 6):
        cell_sub(t6, k, 1, f"{PLAB[k]}\n({ex[k]})")
        cell_sub(t6c, k, 0, PLAB[k])
    cell_sub(t6c, 0, 0, "Profile")
    cell_sub(t6, 2, 6, "Security-integrated microplanning, negotiated access, mobile outreach teams and defaulter tracing")
    cell_sub(t6, 3, 6, "Outreach and mobile sessions (boat-based where waterways apply), community health worker networks, and "
                       "multi-antigen bundling per visit")
    cell_sub(t6, 4, 5, "Near the national average on every indicator; fair care-seeking (ANC 72%, delivery 61%)")
    cell_sub(t6, 4, 6, "Core fixed-site and outreach services, ward enumeration, and Periodic Intensification of Routine Immunization")
    cell_sub(t6, 5, 5, "Highest wealth and education (~9.6 yrs) and facility delivery (78%); moderate conflict events")
    cell_sub(t6c, 2, 6, "Security-integrated microplanning, negotiated access, and mobile outreach teams delivering immunization and "
                        "antenatal care together")
    cell_sub(t6c, 3, 6, "Outreach and mobile sessions (boat-based where waterways apply), community health worker networks, and "
                        "multi-antigen and antenatal bundling per visit")
    cell_sub(t6c, 4, 6, "Ward enumeration and Periodic Intensification of Routine Immunization linked to antenatal clinics")

    # abbreviations
    ab = Tt[0]._tbl
    last = ab.findall(qn("w:tr"))[-1]
    for vals in (("ARI", "Adjusted Rand index", "CRPS", "Continuous ranked probability score"),
                 ("MAE", "Mean absolute error", "", "")):
        tr = copy.deepcopy(last)
        for tc, v in zip(tr.findall(qn("w:tc")), vals):
            set_cell(tc, v)
        last.addnext(tr); last = tr

    # ---- 5.5.6 and 5.5.7, inserted before 5.6 Domain 6
    c = Cursor(doc, P[314]._p.getprevious(), T)
    a1, a4 = TA["A1 External validation"], TA["A4 Targeting capture"]
    c.para("h2", "5.5.6 Validation, Calibration and Uncertainty-Aware Prioritization")
    c.para("body", "Method 2 was tested against the NmDHS 2025-26 state estimates, which were not used to fit any model, alongside "
                   "four alternative approaches. It had the lowest mean absolute error (MAE) of the approaches tested: 5.9 percentage "
                   "points, with near-zero bias (+0.1) and a rank correlation of 0.87 (95% CI 0.73 to 0.95). Method 1 was close "
                   "(MAE 6.3; difference 0.4 points, 95% CI -0.8 to 1.6), so the two methods are statistically equivalent at state "
                   "level. Gradient boosting (7.5) was less accurate, and a Fay-Herriot area-level model (7.9) and the latest survey "
                   "carried forward (8.2) were less accurate with intervals that just include zero. Zero-dose derived from routine "
                   "administrative data alone was far less accurate (MAE 26.5).")
    ren = {"Bayesian small-area estimation (SAE)": M2N, "Hierarchical state model, routine-share allocation": M1N}
    rows = [["Approach", "MAE", "Bias", "Spearman ρ (95% CI)", "CRPS", "MAE difference from Method 2 (95% CI)"]]
    for r in a1.itertuples(index=False):
        rows.append([ren.get(r[0], r[0]), f"{r[1]:.1f}", f"{r[2]:+.1f}", r[3], r[4], r[5]])
    c.table("Table 6i. External validation against the NmDHS 2025-26, 37 states.", rows, [2.05, 0.5, 0.5, 1.07, 0.5, 1.65],
            "Model estimate. MAE and bias in percentage points of state zero-dose prevalence; CRPS = continuous ranked "
            "probability score (lower is better); CI = bootstrap confidence interval over states. The NmDHS 2025-26 was not "
            "used in fitting.")
    c.figure(a_fig(2), "Figure 16k. Comparative performance: state-level error against the NmDHS 2025-26, paired differences "
                       "from Method 2, and survey-round holdouts.")
    c.para("body", "The uncertainty intervals are well calibrated. Allowing for survey sampling error with a design effect of 2, the "
                   "95% predictive interval contained the NmDHS value in 37 of 37 states and the 50% interval in 24 of 37, consistent "
                   "with nominal coverage (p = 0.10 and 0.26). The narrower credible interval for the underlying rate, which excludes "
                   "sampling error, covered 57% and 95%. The continuous ranked probability score (CRPS), which rewards both accuracy "
                   "and sharpness, was 4.4 for Method 2 and 4.5 for Method 1. Within states, Method 2 LGA estimates agree with the "
                   "independently modelled IHME 2018 LGA estimates (within-state rank correlation 0.54, positive in 36 of 37 "
                   "states), compared with 0.08 for allocation by routine-data shares and 0.04 for a model without covariates.")
    c.figure(a_fig(4), "Figure 16l. Calibration: empirical against nominal coverage of credible and predictive intervals, by design "
                       "effect, for both methods.", 5.8)
    c.para("body", "Projecting ahead is harder than describing the present. When survey rounds were held out, Method 2 predicted 2018 "
                   "better than carrying the 2013 survey forward (MAE 10.1 vs 12.0) but predicted 2024 less well than carrying 2018 "
                   "forward (10.9 vs 8.9). In 18 states the model projected the wrong direction of change between 2018 and 2024, and "
                   "errors there were nearly twice as large (MAE 14.3 vs 7.7). The 2026 estimates therefore carry real trend "
                   "uncertainty and should be refreshed whenever a new survey round becomes available.")
    c.figure(a_fig(5), "Figure 16m. Temporal extrapolation: observed against model-implied change from the last training round; "
                       "red marks a wrong projected direction.", 5.8)
    c.para("body", "Uncertainty matters most for targeting. Across 4,000 posterior draws, 40 LGAs are near-certain members of the 155 "
                   "highest-burden LGAs (probability 0.9 or more), 517 are unlikely members (0.1 or less) and 217 are uncertain, of "
                   "which 102 fall outside the ranked list of 155. Ranking by Method 2 expected burden captures 54% of modelled "
                   "zero-dose children with 155 LGAs, compared with 48% when ranking by rate alone and 30 to 34% when ranking by "
                   "routine administrative coverage. Each model scores its own list highest, so these capture figures are model-based "
                   "rather than observed outcomes.")
    hdr = ["Priority rule", "50 LGAs", "100 LGAs", "155 LGAs", "155 LGAs, scored under Method 1", "200 LGAs"]
    rr = {"State-model expected burden": "Method 1 expected burden", "SAE expected burden": "Method 2 expected burden",
          "SAE prevalence (rate) only": "Method 2 rate only", "SAE probability of top-155 membership":
          "Method 2 probability of top-155 membership"}
    rows = [hdr] + [[rr.get(r[0], r[0])] + [f"{x:.1f}" for x in r[1:]] for r in a4.itertuples(index=False)]
    c.table("Table 6j. Share of modelled zero-dose children (%) captured by priority lists of different sizes.", rows,
            [2.17, 0.75, 0.75, 0.75, 1.1, 0.75],
            "Model estimate. Shares are scored under the Method 2 posterior unless stated. The upper bound selects the true "
            "highest-burden LGAs in each posterior draw. Model-based, not observed outcomes.")
    c.figure(a_fig(6), "Figure 16n. Share of modelled zero-dose children captured by priority lists of 25 to 200 LGAs, scored under "
                       "each model's posterior.", 5.8)
    c.para("body", "For planning, the 40 near-certain LGAs should be funded first, and uncertain LGAs should be verified through "
                   "microplanning or rapid assessment before full investment. The results are robust: across design-effect, prior and "
                   "spatial specifications the state MAE ranges from 5.86 to 6.02. Updating the denominator to the 2025 under-five "
                   "projection raises the national total by 12.7% but changes only 4 of the 155 priority LGAs.")

    b2 = TB["B2 Profiles"]
    c.para("h2", "5.5.7 Contextual Profiles and Candidate Intervention Packages")
    c.para("body", "The five community archetypes in Section 5.5.1 were re-examined as contextual profiles: descriptive groupings of "
                   "LGAs by fifteen indicators in six domains (socioeconomic, maternal care, nutrition, access, insecurity, and "
                   "programme performance measured as DTP1-3 dropout). The labels now describe only what the indicators measure: "
                   "deprived northern rural; low maternal care with high conflict exposure; geographically isolated; near-average; "
                   "and relatively advantaged. Earlier labels referring to nomadism, migration, riverine settlement and urban slums "
                   "were retired because no input measures those attributes. Population mobility in particular must be established "
                   "locally before transit-point strategies are considered.")
    c.para("body", "Five profiles are an operational choice rather than a statistical optimum. Four profiles separate the data more "
                   "cleanly (silhouette 0.31 vs 0.18) by merging the near-average and relatively advantaged groups, but those two "
                   "groups differ in burden (16.3% vs 4.5% of modelled zero-dose children) and in dominant barrier. Assignments are "
                   "moderately stable (bootstrap adjusted Rand index (ARI) 0.54, 95% interval 0.42 to 0.72), and the profiles are "
                   "geographically coherent: 79% of neighbouring LGAs share a profile, against 26% expected by chance. Membership is "
                   "clear for 46% of LGAs (60% of modelled zero-dose children), intermediate for 31% and mixed for 23%.")
    dom = ["Socioeconomic", "Maternal care", "Nutrition", "Access", "Insecurity", "Programme (dropout)"]
    rows = [["Profile", "LGAs", "Share of children (%)", "Share of zero-dose, Method 2 (%)", "Mean rate (%)",
             "Domains at or above +0.5 SD", "Candidate components (centroid rule)"]]
    for r in b2.itertuples(index=False):
        d = r._asdict()
        hi = [x.replace("Programme (dropout)", "DTP1-3 dropout").lower() for x in dom if b2.loc[b2.Profile == r.Profile, x].iloc[0] >= 0.5]
        rows.append([r.Profile, r.LGAs, f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.1f}", ", ".join(hi).capitalize().replace("dtp1-3", "DTP1-3") if hi else "None",
                     str(b2.loc[b2.Profile == r.Profile].iloc[0, -1]).capitalize()])
    c.table("Table 6k. Contextual profiles: size, share of children and of modelled zero-dose children, and main barriers.", rows,
            [1.3, 0.45, 0.65, 0.75, 0.55, 1.2, 1.37],
            "Model estimate. Ward hierarchical clustering of 15 indicators, 774 LGAs. Domain scores are standardized means "
            "(higher = more disadvantaged); SD = standard deviation. Shares use the Method 2 zero-dose estimates. A common core of "
            "fixed-site and outreach routine immunization with microplanning applies to every LGA.")
    c.figure(b_fig(2), "Figure 16o. Mean domain score of each contextual profile across six domains (higher = more disadvantaged).", 5.8)
    c.para("body", "Contextual barriers vary within states as much as between them for access and insecurity: 56% of the variance in "
                   "access and 59% in insecurity lies within states, compared with 22% for maternal care and 5% for nutrition. "
                   "Thirty-five of the 37 states contain LGAs with different dominant barriers. Among LGA pairs with nearly identical "
                   "estimated zero-dose rates (within one point), 49.6% belong to different profiles and 67% have a different "
                   "dominant barrier, although such pairs are contextually closer than random pairs (median distance 1.8 vs 2.8). "
                   "The same zero-dose rate can therefore reflect different barriers in a substantial minority of LGAs, which a "
                   "rate-only ranking cannot show.")
    c.figure(b_fig(3), "Figure 16p. Dominant contextual barrier by LGA, and profile membership classified as clear, intermediate or "
                       "mixed.")
    c.figure(b_fig(4), "Figure 16q. LGA pairs with near-identical estimated zero-dose rates but different dominant barriers.")
    c.para("body", "The two highest-burden profiles hold 77.4% of Method 2 modelled zero-dose children but 51.3% of the birth cohort. "
                   "At state level, the share of children living in these two profiles improves prediction of the NmDHS 2025-26 "
                   "beyond the NDHS 2024 (leave-one-out R-squared 0.71 to 0.78); the NmDHS was not used to build any indicator. "
                   "Out of sample at LGA level, profiles add information beyond geopolitical zone (R-squared gain +0.08) but little "
                   "beyond zone and a deprivation quintile (+0.01), partly because they share inputs with the estimation models. "
                   "Profiles are therefore a summary for planning, not an independent predictor of zero-dose.")
    c.para("body", "Candidate packages are assigned from each LGA's own flagged barriers using documented rules (Table 6l): a barrier "
                   "is flagged when its domain score is in the national top quartile, or travel time is in the top decile. Of the 774 "
                   "LGAs, 36%, holding 67% of modelled zero-dose children, have two or more flagged barriers, so combined packages are "
                   "the norm where burden is highest.")
    rows = [["Barrier", "Flagging rule", "Candidate components", "Evidence or guideline", "Local validation required"]]
    for r in BMAT.itertuples(index=False):
        rows.append([r.barrier.replace("Programme (dropout)", "Programme performance (DTP1-3 dropout)"), r.rule, r.components,
                     r.evidence, r.local_validation])
    c.table("Table 6l. Barrier-to-intervention mapping: flagging rules, candidate components and local-validation requirements.",
            rows, [1.0, 1.25, 1.45, 1.4, 1.17],
            "Rules are applied to each LGA's domain scores. Components are candidates for local validation, not measured "
            "intervention effects. Population mobility has no input indicator and is never assigned from these data.")
    c.para("body", "How packages are assigned matters. Measured against each LGA's barrier-based package (burden-weighted Jaccard "
                   "similarity, where 1 means identical), assigning by state agrees best (0.65), followed by contextual profile "
                   "(0.58), profile plus secondary profile (0.53), geopolitical zone (0.42), deprivation quintile (0.41), zero-dose "
                   "rate quintile (0.40) and one national package (0.13). State packages refined with each LGA's flagged barriers are "
                   "therefore the most consistent with the data, and a single national package the least. These are agreements with "
                   "documented rules, not measured effects of interventions.")
    c.figure(b_fig(5), "Figure 16r. Share of each domain's variance lying within states, and agreement of package-assignment rules "
                       "with each LGA's barrier package.")
    c.para("body", "Core package components are retained for 51 to 82% of LGAs (71 to 96% of modelled zero-dose children) across "
                   "alternative indicator sets, numbers of profiles, clustering algorithms and bootstrap refits, and for 73 to 82% of "
                   "LGAs when the barrier-flagging threshold is changed. The LGA barrier flags do not depend on the clustering at all. "
                   "Packages should be validated locally through microplanning before adoption.")
    c.figure(b_fig(6), "Figure 16s. Stability of candidate packages across indicator sets, numbers of profiles, algorithms and "
                       "bootstrap refits.", 5.6)

    doc.save(str(OUT_DOC))
    print("report saved", OUT_DOC.name)


# ================================================================== DECK
from lxml import etree  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn as pqn  # noqa: E402
from pptx.util import Inches as In, Pt  # noqa: E402

NAVY, INK, SUB, FOOT, GREEN = "1F3B57", "22333F", "45596B", "5C6B78", "1C7A3D"
LIGHT, LINE, GTINT, GLINE, WHITE = "EEF2F6", "D6DEE7", "E7F1EB", "C9D6E2", "FFFFFF"


def rgb(h):
    return RGBColor.from_string(h)


def shp(slide, name):
    hits = [s for s in slide.shapes if s.name == name]
    assert len(hits) == 1, (name, len(hits))
    return hits[0]


def set_text(sh, paras):
    if isinstance(paras, str):
        paras = [[paras]]
    elif paras and not isinstance(paras[0], list):
        paras = [paras]
    txb = sh.text_frame._txBody
    ps = txb.findall(pqn("a:p"))
    runs = [r for p in ps for r in p.findall(pqn("a:r"))]
    bold_t = next((r for r in runs if r.find(pqn("a:rPr")) is not None and r.find(pqn("a:rPr")).get("b") == "1"), None)
    reg_t = next((r for r in runs if r.find(pqn("a:rPr")) is None or r.find(pqn("a:rPr")).get("b") != "1"), None)
    first = runs[0]
    p0 = copy.deepcopy(ps[0])
    for ch in list(p0):
        if ch.tag in (pqn("a:r"), pqn("a:br"), pqn("a:fld")):
            p0.remove(ch)
    for p in ps:
        txb.remove(p)
    for spec in paras:
        p = copy.deepcopy(p0)
        end = p.find(pqn("a:endParaRPr"))
        for run in spec:
            txt, b = (run, None) if isinstance(run, str) else run
            tmpl = first if b is None else (bold_t if b else reg_t)
            r = copy.deepcopy(tmpl if tmpl is not None else first)
            r.find(pqn("a:t")).text = txt
            rpr = r.find(pqn("a:rPr"))
            if b is not None and rpr is not None:
                rpr.set("b", "1" if b else "0")
            if end is not None:
                end.addprevious(r)
            else:
                p.append(r)
        txb.append(p)


def fit(png, w, h):
    iw, ih = Image.open(png).size
    ar = iw / ih
    return (w, w / ar) if w / ar <= h else (h * ar, h)


def replace_picture(slide, old, png):
    x, y, w, h = old.left / 914400, old.top / 914400, old.width / 914400, old.height / 914400
    fw, fh = fit(png, w, h)
    pic = slide.shapes.add_picture(str(png), In(x + (w - fw) / 2), In(y + (h - fh) / 2), In(fw), In(fh))
    old._element.addprevious(pic._element)
    old._element.getparent().remove(old._element)


def picture(slide, png, x, y, w, h):
    fw, fh = fit(png, w, h)
    return slide.shapes.add_picture(str(png), In(x + (w - fw) / 2), In(y), In(fw), In(fh))


def rect(slide, x, y, w, h, fill=LIGHT, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, In(x), In(y), In(w), In(h))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, paras, size=14, color=INK, bold=False, italic=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    s = slide.shapes.add_textbox(In(x), In(y), In(w), In(h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for run in ([para] if isinstance(para, str) else para):
            t, o = (run, {}) if isinstance(run, str) else run
            r = p.add_run(); r.text = t
            f = r.font; f.name = "Arial"; f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold); f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return s


def header(slide, title, subtitle=None):
    text(slide, 0.60, 0.26, 12.13, 0.57, title, size=28, color=NAVY, bold=True)
    if subtitle:
        text(slide, 0.60, 0.87, 12.13, 0.34, subtitle, size=16, color=SUB)


def footer(slide, s):
    text(slide, 0.60, 6.80, 12.13, 0.45, s, size=10, color=FOOT, italic=True)


def kpi(slide, x, y, w, h, big, label, big_size=26, label_size=12):
    rect(slide, x, y, w, h, GTINT, GLINE)
    text(slide, x + 0.18, y + 0.1, w - 0.36, 0.5, big, size=big_size, color=GREEN, bold=True)
    text(slide, x + 0.18, y + 0.1 + big_size / 72 * 1.3, w - 0.36, h - 0.25 - big_size / 72 * 1.3, label,
         size=label_size, color=NAVY, bold=True)


def card(slide, x, y, w, h, lead, body, size=12, fill=LIGHT, line=LINE):
    rect(slide, x, y, w, h, fill, line)
    text(slide, x + 0.15, y + 0.09, w - 0.3, h - 0.18, [[(lead + " ", {"bold": True, "color": NAVY}), body]], size=size)


def new_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    return s


PFOOT = ("Model estimate. Ward hierarchical clustering of 15 indicators in six domains (socioeconomic, maternal care, nutrition, "
         "access, insecurity, DTP1-3 dropout), 774 LGAs. ARI = adjusted Rand index; SD = standard deviation.")


def build_deck():
    prs = Presentation(str(SRC_PPT))
    S = list(prs.slides)
    assert len(S) == 46
    ids = list(prs.slides._sldIdLst)
    anchor22, anchor35 = ids[21], ids[34]

    # ---- relabel existing slides
    set_text(shp(S[1], "TextBox 19"), "All 774 LGAs grouped into five contextual profiles from 15 local indicators, using no "
                                      "zero-dose or coverage estimate, with candidate packages from each LGA's measured barriers. "
                                      "Two profiles hold 1.52 million missed children.")
    s = S[18]
    set_text(shp(s, "Text 0"), "Contextual profiles and modelled zero-dose burden")
    set_text(shp(s, "Text 1"), "Built from 15 contextual indicators (education, nutrition, poverty, wealth, travel time, water, maternal "
                               "care, conflict and DTP1-3 dropout), the profiles use no zero-dose or coverage estimate. Yet they "
                               "reproduce where the children are: profiles 1 and 2 hold 1.52 million of 2.10 million zero-dose "
                               "children (72 percent; 77 percent under Method 2).")
    set_text(shp(s, "Text 2"), "Colour shows the contextual profile. Bars compare each profile's share of children with its share of "
                               "zero-dose children under both methods. Labels describe only what the indicators measure.")
    set_text(shp(s, "Text 3"), "Model estimate. Profiles (community archetypes) from agglomerative (Ward) clustering of 15 LGA "
                               "indicators (IHME, DHS, Meta Relative Wealth Index, Weiss travel time, ACLED); bars are modelled zero-dose "
                               "children under both methods; GRID3 admin-2 boundaries.")
    replace_picture(s, shp(s, "Picture 7"), FIG / "P_01_profile_map_burden.png")

    s = S[19]
    set_text(shp(s, "Text 0"), "Five contextual profiles and candidate responses")
    set_text(shp(s, "Text 1"), "Profiles 1 and 2 hold about 72 percent of all zero-dose children (77 percent under Method 2).")
    set_text(shp(s, "Text 4"), "Profile")
    for k, nm in zip(range(1, 6), ("Text 11", "Text 20", "Text 28", "Text 37", "Text 45")):
        set_text(shp(s, nm), PLAB[k])
    set_text(shp(s, "Text 26"), "Security-integrated microplanning, negotiated access, mobile outreach teams and defaulter tracing")
    set_text(shp(s, "Text 34"), "Outreach and mobile sessions (boat-based where waterways apply), community health workers, "
                                "multi-antigen bundling")
    set_text(shp(s, "Text 42"), "Near the national average on every indicator; fair care-seeking (antenatal care 72%, delivery 61%)")
    set_text(shp(s, "Text 43"), "Core fixed-site and outreach services, ward enumeration, and Periodic Intensification of Routine "
                                "Immunization")
    set_text(shp(s, "Text 50"), "Highest wealth and education (about 9.6 years) and facility delivery (78%); moderate conflict events")
    set_text(shp(s, "Text 52"), "Model estimate. Ward clustering of 15 indicators across all 774 LGAs; labels describe measured "
                                "indicators only. Zero-dose children and mean LGA rate from Method 1 (total 2.10 million). Grey text "
                                "names example states. BHCPF = Basic Health Care Provision Fund.")

    s = S[20]
    set_text(shp(s, "Text 4"), "of the 2.10 million modelled zero-dose children (Method 1) are in the two profiles with the weakest "
                               "maternal care.")
    set_text(shp(s, "Text 11"), [[("The burden. ", True), ("The deprived northern rural and low maternal care, high conflict exposure "
                                 "profiles hold about 1.5 million zero-dose children, on the lowest antenatal care (36 to 40%) and "
                                 "facility delivery (10 to 19%).", False)]])
    set_text(shp(s, "Text 13"), [[("The action. ", True), ("Co-deliver immunization with antenatal and delivery contacts in these "
                                 "profiles: one visit, several services, rather than immunization outreach on its own.", False)]])
    set_text(shp(s, "Text 14"), "Model estimate. Source: LGA profile master (IHME and DHS antenatal-care and facility-delivery "
                                "covariates; modelled zero-dose). RMNCH = reproductive, maternal, newborn and child health; EPI = "
                                "Expanded Programme on Immunization; ANC4+ = antenatal care, four or more visits; LGA = local "
                                "government area.")

    s = S[21]
    set_text(shp(s, "Text 0"), "From profile to action: the RMNCH-EPI operational list")
    set_text(shp(s, "Text 1"), "Five contextual profiles, the maternal-care numbers behind each, and the candidate package.")
    set_text(shp(s, "Text 3"), "Profile")
    set_text(shp(s, "Text 9"), "RMNCH-EPI intervention, matched to the profile")
    for k, nm in zip(range(1, 6), ("Text 11", "Text 18", "Text 26", "Text 33", "Text 41")):
        set_text(shp(s, nm), PLAB[k])
    set_text(shp(s, "Text 24"), "Security-integrated microplanning, negotiated access, and mobile outreach teams delivering "
                                "immunization and antenatal care together")
    set_text(shp(s, "Text 32"), "Outreach and mobile sessions (boat-based where waterways apply), community health workers, and "
                                "multi-antigen and antenatal bundling per visit")
    set_text(shp(s, "Text 39"), "Ward enumeration and Periodic Intensification of Routine Immunization linked to antenatal clinics")
    set_text(shp(s, "Text 49"), [[("The full list. ", True), ("All 774 local government areas are ranked worst-to-best by modelled "
                                 "zero-dose burden in the operational workbook, each carrying its profile, maternal-care figures and "
                                 "candidate package, ready for microplanning.", False)]])

    s = S[39]
    set_text(shp(s, "Text 9"), "Barrier-matched packages")
    set_text(shp(s, "Text 10"), "Use each LGA's measured barriers to choose the package, starting from the state package.")
    s = S[43]
    set_text(shp(s, "Text 9"), "Match packages to each LGA's measured barriers")
    set_text(shp(s, "Text 10"), "Start from the state package and add outreach, security-sensitive access, maternal-care or nutrition "
                                "integration, demand support or defaulter tracing where an LGA's barriers call for it; validate locally.")

    # ---- profile slides (after slide 22)
    P_NEW = []
    s = new_slide(prs); P_NEW.append(s)
    header(s, "Five contextual profiles: what sets them apart",
           "Mean domain scores (higher = more disadvantaged); labels describe only what the indicators measure.")
    picture(s, b_fig(2), 0.60, 1.35, 8.0, 3.6)
    card(s, 8.85, 1.35, 3.88, 1.15, "An operational choice.", "Four profiles fit slightly better (silhouette 0.31 vs 0.18); five are "
         "kept because near-average and relatively advantaged LGAs differ in burden (16.3% vs 4.5%).")
    card(s, 8.85, 2.58, 3.88, 1.1, "Coherent.", "79% of neighbouring LGAs share a profile (26% by chance); bootstrap stability "
         "ARI 0.54 (0.42 to 0.72).")
    card(s, 8.85, 3.76, 3.88, 1.19, "Membership.", "Clear for 46% of LGAs (60% of modelled zero-dose children), intermediate for "
         "31%, mixed for 23%.")
    card(s, 0.60, 5.15, 12.13, 1.35, "Labels retired.", "Earlier names for nomadic, migrant, riverine and urban-slum areas were "
         "dropped because no indicator measures those attributes. Population mobility must be confirmed locally before transit-point "
         "strategies are used. The two highest-burden profiles hold 77.4% of Method 2 modelled zero-dose children but 51.3% of the "
         "birth cohort.", size=13, fill=GTINT, line=GLINE)
    footer(s, PFOOT)

    s = new_slide(prs); P_NEW.append(s)
    header(s, "Barriers differ within states and at the same zero-dose rate",
           "Dominant barrier by LGA, and how clearly each LGA fits its profile.")
    picture(s, b_fig(3), 0.60, 1.35, 8.2, 3.7)
    kpi(s, 9.05, 1.35, 3.68, 1.17, "35 of 37", "states contain LGAs with different dominant barriers")
    kpi(s, 9.05, 2.6, 3.68, 1.17, "56% / 59%", "of access and insecurity variance lies within states (nutrition 5%)")
    kpi(s, 9.05, 3.85, 3.68, 1.2, "67%", "of LGA pairs with near-equal rates have a different dominant barrier")
    card(s, 0.60, 5.2, 12.13, 1.4, "Caveat.", "Near-equal pairs are still contextually closer than random pairs (median distance 1.8 vs "
         "2.8), so 'same rate, different barriers' applies to a substantial minority, not to every LGA. State averages capture "
         "nutrition and maternal care well but hide access and insecurity, which need LGA-level planning.", size=13)
    footer(s, PFOOT + " Dominant barrier = highest domain score at or above +0.5 SD.")

    s = new_slide(prs); P_NEW.append(s)
    header(s, "Same estimated rate, different barriers", "Four real LGA pairs with near-identical Method 2 zero-dose rates.")
    picture(s, b_fig(4), 0.60, 1.35, 12.13, 3.6)
    card(s, 0.60, 5.1, 5.97, 1.5, "Reading the pairs.", "Each pair has nearly the same estimated zero-dose rate but a different "
         "dominant barrier, so a rate-only ranking would send both the same response. Across all near-equal pairs, 49.6% fall in "
         "different profiles.", size=13)
    card(s, 6.76, 5.1, 5.97, 1.5, "Caution.", "All four pairs contrast Niger Delta with Jigawa LGAs, and their rate intervals are "
         "wide. They illustrate the pattern; they do not measure how common it is.", size=13)
    footer(s, PFOOT)

    s = new_slide(prs); P_NEW.append(s)
    header(s, "Assigning packages: start from the state, refine by LGA barriers",
           "Agreement of each assignment rule with every LGA's own barrier package, weighted by modelled zero-dose children.")
    picture(s, b_fig(5), 0.60, 1.35, 12.13, 3.85)
    for i, (lead, body) in enumerate([
            ("Combined barriers are the norm.", "36% of LGAs, holding 67% of modelled zero-dose children, have two or more flagged "
             "barriers."),
            ("State first, then LGA.", "State packages agree best (0.65), profiles next (0.58); one national package agrees least "
             "(0.13)."),
            ("Validate locally.", "Packages are candidate components from documented rules, not measured intervention effects.")]):
        card(s, 0.60 + i * 4.11, 5.3, 3.91, 1.35, lead, body, size=12.5)
    footer(s, "Model estimate. Barrier flag = domain score in the national top quartile (travel time: top decile). Agreement = "
              "Jaccard similarity (1 = identical). Population mobility is not measured and is never assigned.")

    s = new_slide(prs); P_NEW.append(s)
    header(s, "How robust are the candidate packages?", "Share of LGAs and children keeping the same core components when modelling "
           "choices change.")
    picture(s, b_fig(6), 0.60, 1.35, 7.0, 5.25)
    for i, (lead, body, fill, line) in enumerate([
            ("Stable core.", "Core components are kept for 51-82% of LGAs (71-96% of modelled children) across indicator sets, "
             "numbers of profiles, algorithms and bootstrap refits.", LIGHT, LINE),
            ("Threshold.", "Changing the barrier-flagging threshold keeps 73-82% of LGA packages; LGA flags do not depend on the "
             "clustering.", LIGHT, LINE),
            ("Information content.", "Profiles add to zone in explaining LGA zero-dose (R-squared +0.08) but little beyond a "
             "deprivation quintile (+0.01); at state level they improve prediction of the NmDHS 2025-26 (R-squared 0.71 to 0.78).",
             LIGHT, LINE),
            ("Use.", "Choose components from LGA barrier flags; use profiles as a planning summary, not as an independent "
             "predictor.", GTINT, GLINE)]):
        card(s, 7.85, 1.35 + i * 1.33, 4.88, 1.23, lead, body, size=11.5, fill=fill, line=line)
    footer(s, PFOOT + " Core components = Jaccard similarity of 0.67 or more; R-squared out of sample (state check leave-one-out).")

    # ---- validation slides (after slide 35)
    V_NEW = []
    s = new_slide(prs); V_NEW.append(s)
    header(s, "Method 2 is the most accurate approach tested", "Independent state-level check against the NmDHS 2025-26 "
           "(37 states); the survey was not used to fit any model.")
    picture(s, a_fig(2), 0.60, 1.35, 12.13, 3.7)
    kpi(s, 0.60, 5.2, 3.91, 1.45, "5.9 points", "mean absolute error (MAE), lowest of six approaches; bias +0.1")
    kpi(s, 4.71, 5.2, 3.91, 1.45, "0.87", "rank correlation with the survey (95% CI 0.73 to 0.95)")
    kpi(s, 8.82, 5.2, 3.91, 1.45, "+0.4 points", "Method 1 difference (95% CI -0.8 to 1.6): statistically equivalent")
    footer(s, "Model estimate. MAE in percentage points; CI = bootstrap confidence interval over states. Comparators: Method 1 (6.3), "
              "gradient boosting (7.5), Fay-Herriot area-level model (7.9), latest survey carried forward (8.2), routine data only (26.5).")

    s = new_slide(prs); V_NEW.append(s)
    header(s, "Honest uncertainty; forecasting ahead is the weak link",
           "Intervals cover the survey as intended; projecting trends forward is harder than describing the present.")
    picture(s, a_fig(4), 0.60, 1.35, 5.95, 2.45)
    picture(s, a_fig(5), 6.78, 1.35, 5.95, 2.45)
    card(s, 0.60, 3.95, 5.95, 1.6, "Calibrated.", "With design effect 2, the 95% predictive interval contained the NmDHS value in 37 "
         "of 37 states and the 50% interval in 24 of 37, as expected (p = 0.10 and 0.26). Continuous ranked probability score (CRPS) "
         "4.4, against 4.5 for Method 1.")
    card(s, 6.78, 3.95, 5.95, 1.6, "Trend breaks.", "With survey rounds held out, Method 2 beat carry-forward for 2018 (MAE 10.1 vs "
         "12.0) but not for 2024 (10.9 vs 8.9). In 18 states the projected direction was wrong, with errors nearly twice as large "
         "(14.3 vs 7.7).")
    card(s, 0.60, 5.7, 12.13, 0.95, "What it means.", "Use the intervals, not only the point estimates, and refresh the model with "
         "every new survey round: the 2026 estimates carry real trend uncertainty.", size=13, fill=GTINT, line=GLINE)
    footer(s, "Model estimate. Design effect 2 allows for survey sampling error. MAE = mean absolute error, percentage points.")

    s = new_slide(prs); V_NEW.append(s)
    header(s, "Prioritizing under uncertainty", "Fund near-certain priorities first; verify uncertain LGAs through microplanning.")
    picture(s, a_fig(6), 0.60, 1.35, 7.6, 3.05)
    kpi(s, 8.5, 1.35, 4.23, 1.4, "40 LGAs", "near-certain to be in the top 155 (probability 0.9 or more)")
    kpi(s, 8.5, 2.85, 4.23, 1.4, "217 LGAs", "uncertain; 102 of them fall outside the ranked list of 155")
    kpi(s, 8.5, 4.35, 4.23, 1.4, "54%", "of modelled zero-dose children in the 155 LGAs ranked by Method 2 expected burden")
    card(s, 0.60, 4.55, 7.6, 1.2, "Capture.", "Ranking by rate alone captures 48%, and ranking by routine administrative coverage "
         "30-34%. Each model favours its own list, so these are model-based, not observed.")
    card(s, 0.60, 5.85, 12.13, 0.85, "Robust.", "State MAE stays at 5.86-6.02 across design-effect, prior and spatial choices; the "
         "2025 population projection raises the total by 12.7% but changes only 4 of the 155 priority LGAs.", size=12.5,
         fill=GTINT, line=GLINE)
    footer(s, "Model estimate. Method 2: Bayesian small-area estimation (SAE), 4,000 posterior draws. MAE = mean absolute error.")

    # ---- reorder
    lst = prs.slides._sldIdLst
    all_ids = list(lst)
    p_ids, v_ids = all_ids[46:46 + len(P_NEW)], all_ids[46 + len(P_NEW):]
    for e in p_ids + v_ids:
        lst.remove(e)
    for anchor, new in ((anchor22, p_ids), (anchor35, v_ids)):
        for e in reversed(new):
            anchor.addnext(e)
    prs.save(str(OUT_PPT))
    print("deck saved", OUT_PPT.name, len(prs.slides), "slides")


if __name__ == "__main__":
    build_report()
    build_deck()
