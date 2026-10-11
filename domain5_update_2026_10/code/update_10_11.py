"""Build the 11.10.2026 report and deck: Domain 2 rebuilt with correctly parsed DHIS2 counts, plus the archetype title.

Starts from the 10.10.2026 report and deck (never modified). Domain 2 figures come from
Domain2_Update_20261011/code/d2_rebuild.py, which uses the web app's own parsing, Prophet and LASSO functions.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from docx import Document

import update_10_10 as U

ROOT = Path(__file__).resolve().parents[1]
RD = ROOT / "report_and_deck"
D2 = ROOT.parent / "Domain2_Update_20261011"
F = D2 / "figures"
S = json.loads((D2 / "results" / "summary.json").read_text())
SRC_DOC = RD / "Final_NPHCDA_Consolidated_Report_Domain5_Two_Methods_10.10.2026.docx"
OUT_DOC = RD / "Final_NPHCDA_Consolidated_Report_Domain5_Two_Methods_10.11.2026.docx"
SRC_PPT = ROOT.parents[1] / "nphcda-zerodose-git" / "domain5_update_2026_10" / "report_and_deck" / \
    "NPHCDA_ZeroDose_RI_Team_Presentation_Domain5_Two_Methods_10.10.2026.pptx"
OUT_PPT = RD / "NPHCDA_ZeroDose_RI_Team_Presentation_Domain5_Two_Methods_10.11.2026.pptx"

NAT = {r["col"]: r for r in S["national"]}
m12 = NAT["dropout_m1m2"]
SY = S["state_year"]["dropout_m1m2"]
TOP5 = SY["top5_2025"]
assert m12["mean_2021"] == 60.4 and m12["mean_2025"] == 33.5 and SY["states_above_10_2025"] == 34
assert list(TOP5) == ["Anambra", "Akwa Ibom", "Imo", "Abia", "Oyo"]
assert S["state_lasso"]["dropout_p1p3"] == {}

P247 = ("The three antigen transitions tell different stories. Penta1-to-Penta3 dropout is low and stable: 6 to 7% a year nationally in "
        "2021-2024 and 4.8% in 2025, with a forecast of about 6.6% for December 2026 (95% prediction interval 2.6 to 10.8%), so most "
        "children who start the Penta series complete it. Penta1-to-Measles1 dropout is moderate, about 11 to 16% a year (10.8% in 2025), "
        "with wide month-to-month swings. The decisive transition is Measles1-to-Measles2: national dropout fell from 60% in 2021 to 34% "
        "in 2025 and is forecast at about 26% by December 2026 (20 to 32%), much improved but still by far the largest of the three gaps. "
        "In 2025, 34 of 37 states lost more than 10% of children between the two measles doses, with the highest rates in Anambra (60%), "
        "Akwa Ibom (58%), Imo (54%), Abia (53%) and Oyo (48%).")
P248 = ("Cross-validated LASSO driver selection points to different mechanisms for each transition, so a single retention strategy will "
        "not fix all three. Across states (2024), no equity covariate consistently predicts Penta1-to-Penta3 dropout; across LGAs "
        "(2021-2024), it is higher where conflict fatalities are higher and lower where oral rehydration salts (ORS) coverage and facility "
        "delivery are higher. Penta1-to-Measles1 dropout is higher where distance to a facility is a problem and fertility is higher, and "
        "lower where four or more antenatal visits are common; across LGAs it rises with child wasting and stunting. Measles1-to-Measles2 "
        "dropout is lower where vaccination cards are seen and births are registered; its positive association with antenatal care reflects "
        "the high second-dose dropout of southern states, where antenatal coverage is also high, rather than a causal effect. These are "
        "ecological associations (37 states; 743 LGAs, R-squared 0.10 to 0.12) that indicate direction rather than size. Retention efforts "
        "therefore need to be tailored by transition: defaulter tracing and security-sensitive outreach for the Penta series in "
        "conflict-affected LGAs; linking the nine-month visit to outreach and nutrition contacts where distance and undernutrition are "
        "barriers; and home-based records, card retention and second-year reminders for the measles second dose, concentrated on the "
        "high-dropout states of the South-East and South-South and on Oyo.")
NOTE = ("Model estimate. DHIS2 monthly counts 2021-2025, read with thousands separators removed (DHIS2 exports counts of 1,000 or more "
        "as text, for example \"1,234\"). Negative dropout means the later dose was recorded at or above the earlier dose in that period. "
        "The same calculations run in the Dropout & Completion page of the web application.")
REC = ("Close the second-measles-dose gap in the high-dropout states of the South-East and South-South and in Oyo, and audit any "
       "negative dropout as a denominator or reporting issue.")


def build_report():
    doc = Document(str(SRC_DOC))
    P = list(doc.paragraphs)
    assert P[246].text.startswith("5.2 Domain 2") and P[250].text.startswith("Figure 2.") and P[254].text.startswith("Figure 4.")
    assert P[585].text.startswith("Close the second-measles-dose leak")
    note_tmpl = next(p for p in P if p.text.startswith("Model estimate, 2026. Method 1: Bayesian hierarchical"))
    T = {"imgp": copy.deepcopy(P[253]._p), "figcap": copy.deepcopy(P[254]._p), "note": copy.deepcopy(note_tmpl._p)}
    for p in T.values():  # drop images from the template paragraph
        pass
    U.set_p(P[247]._p, P247)
    U.set_p(P[248]._p, P248)
    U.replace_image(doc, P[249]._p, F / "D2_F1_dropout_forecasts.png", 6.27)
    U.set_p(P[250]._p, "Figure 2. National dropout rates for the three antigen transitions: observed DHIS2 monthly series and Prophet "
                       "forecast with 80% and 95% prediction intervals.")
    U.replace_image(doc, P[251]._p, F / "D2_F2_dropout_heatmap.png", 6.27)
    U.set_p(P[252]._p, "Figure 3. Dropout by state and year, observed 2021-2025 and Prophet forecast 2026-2027 (Penta1-to-Penta3 and "
                       "Measles1-to-Measles2).")
    U.replace_image(doc, P[253]._p, F / "D2_F3_dropout_drivers_state.png", 6.27)
    U.set_p(P[254]._p, "Figure 4. Drivers of dropout across states: signed LASSO coefficients, 2024 (red = associated with higher "
                       "dropout; green = lower).")
    c = U.Cursor(doc, P[254]._p, T)
    c.figure(F / "D2_F4_dropout_drivers_lga.png", "Figure 4b. Drivers of dropout across LGAs: signed LASSO coefficients on 15 LGA "
                                                   "covariates, 2021-2024.")
    c.para("note", NOTE)
    U.set_p(P[585]._p, REC)
    doc.save(str(OUT_DOC))
    print("report saved", OUT_DOC.name)


def build_deck():
    from pptx import Presentation
    prs = Presentation(str(SRC_PPT))
    SL = list(prs.slides)
    assert len(SL) == 54
    s = SL[12]
    assert U.shp(s, "Text 0").text_frame.text.startswith("Domain 2")
    U.set_text(U.shp(s, "Text 1"), "Dropout differs by transition: the second measles dose is still the largest gap, and each "
                                   "transition has different drivers.")
    U.replace_picture(s, U.shp(s, "Image 0"), F / "D2_F3_dropout_drivers_state.png")
    U.set_text(U.shp(s, "Text 2"), "The biggest gap is between Measles1 and Measles2: national dropout fell from 60% in 2021 to 34% in "
                                   "2025, still far above Penta1 to Penta3 (about 5%) and Penta1 to Measles1 (about 11%).")
    U.set_text(U.shp(s, "Text 5"), "Higher with conflict deaths; lower with ORS use (LGAs).")
    U.set_text(U.shp(s, "Text 6"), "Defaulter tracing and security-sensitive outreach in conflict-affected LGAs.")
    U.set_text(U.shp(s, "Text 9"), "Distance to care; high fertility; less antenatal care.")
    U.set_text(U.shp(s, "Text 10"), "Link the 9-month visit to outreach and nutrition contacts.")
    U.set_text(U.shp(s, "Text 13"), "Few vaccination cards seen; births unregistered.")
    U.set_text(U.shp(s, "Text 14"), "Home-based records, card retention and second-year reminders.")
    U.set_text(U.shp(s, "Text 15"), "Model estimate. DHIS2 2021-2025 (counts read with thousands separators). Cross-validated LASSO: 37 states "
                                    "(2024; chart) and 743 LGAs (2021-2024). Ecological associations: directional, not causal. "
                                    "ORS = oral rehydration salts.")

    # new slide: state x year heatmap, inserted after slide 13
    n = U.new_slide(prs)
    U.header(n, "Domain 2: where dropout is highest, by state",
             "Observed 2021-2025 and Prophet forecast 2026-2027, the same view as the web application's heatmap.")
    U.picture(n, F / "D2_F2_dropout_heatmap.png", 0.6, 1.3, 7.6, 5.4)
    U.kpi(n, 8.5, 1.35, 4.23, 1.3, "34%", "Measles1 to Measles2 dropout nationally in 2025 (60% in 2021)")
    U.kpi(n, 8.5, 2.75, 4.23, 1.3, "34 of 37", "states lost more than 10% of children between the two measles doses in 2025")
    U.kpi(n, 8.5, 4.15, 4.23, 1.3, "about 5%", "Penta1 to Penta3 dropout in 2025; low and stable since 2021")
    U.card(n, 8.5, 5.55, 4.23, 1.1, "Act first", "in Anambra, Akwa Ibom, Imo, Abia and Oyo, where 48-60% of children "
                                                 "who receive Measles1 miss Measles2.", size=11.5, fill=U.GTINT, line=U.GLINE)
    U.footer(n, "Model estimate. DHIS2 monthly counts read with thousands separators; grey = no records for the state that year; "
                "negative dropout = later dose recorded at or above the earlier dose.")
    lst = prs.slides._sldIdLst
    ids = list(lst)
    new_id = ids[-1]
    lst.remove(new_id)
    ids[12].addnext(new_id)

    SL = list(prs.slides)
    # archetype title (was slide 19, now 20)
    s = SL[19]
    t0 = U.shp(s, "Text 0")
    assert t0.text_frame.text.startswith("Contextual profiles and modelled")
    U.set_text(t0, "Archetype profiles and modelled zero-dose burden")
    from pptx.util import Pt
    for p in t0.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(26)
    # slide-number references on slide 2
    s2 = SL[1]
    U.set_text(U.shp(s2, "Rounded Rectangle 20"), "New slides 20 to 23")
    U.set_text(U.shp(s2, "Rounded Rectangle 29"), "New slide 19")
    r30 = U.shp(s2, "Rounded Rectangle 30")
    old = r30.text_frame.text
    assert "11, 18, 19, 20, 21 and 22" in old
    for p in r30.text_frame.paragraphs:
        for r in p.runs:
            r.text = r.text.replace("11, 18, 19, 20, 21 and 22", "11, 19, 20, 21, 22 and 23")
    assert "11, 19, 20, 21, 22 and 23" in r30.text_frame.text
    prs.save(str(OUT_PPT))
    print("deck saved", OUT_PPT.name, len(prs.slides), "slides")


if __name__ == "__main__":
    build_report()
    build_deck()
