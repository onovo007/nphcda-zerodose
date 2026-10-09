"""Build the Domain 5 two-method version of the NPHCDA RI team deck.

Reads the 10.6.2026 deck (never modified), updates Domain 5 numbers and figures to the canonical results
(results/*.csv, results.json), adds the Method 2 (Bayesian small-area estimation) and comparison slides,
and writes a new file in report_and_deck/.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pandas as pd
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parents[0].parent / "Gavi Modeling Overview" / "NPHCDA_ZeroDose_RI_Team_Presentation_ED_Updates.10.6.2026.pptx"
OUT = ROOT / "report_and_deck" / "NPHCDA_ZeroDose_RI_Team_Presentation_Domain5_Two_Methods_10.9.2026.pptx"
FIG, DFIG, RES = ROOT / "figures", ROOT / "figures" / "deck", ROOT / "results"

NAVY, INK, SUB, FOOT, GREEN = "1F3B57", "22333F", "45596B", "5C6B78", "1C7A3D"
LIGHT, LINE, GTINT, GLINE, WHITE = "EEF2F6", "D6DEE7", "E7F1EB", "C9D6E2", "FFFFFF"
M1C, M2C = "0072B2", "D55E00"
M1N = "Method 1: Bayesian hierarchical model with DHIS2-calibrated LGA allocation"
M2N = "Method 2: Bayesian small-area estimation (SAE)"

# ------------------------------------------------------------------ data
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
m1, m2 = R["m1"], R["m2"]
nat = ZN[ZN.area == "National"].iloc[0]


def mil(x, d=2):
    return f"{x / 1e6:.{d}f} million"


def k(x):
    return f"{x:,.0f}"


def v(method, check):
    r = VAL[(VAL.method == method) & (VAL.check.str.startswith(check))]
    assert len(r) == 1, (method, check)
    return r.iloc[0]


v1n, v2n = v("Method 1", "NmDHS"), v("Method 2 (SAE)", "NmDHS")
v1i, v2i = v("Method 1", "IHME DTP1 2018, LGA overall"), v("Method 2 (SAE)", "IHME DTP1 2018, LGA overall")
v1w, v2w = v("Method 1", "IHME DTP1 2018, LGA within"), v("Method 2 (SAE)", "IHME DTP1 2018, LGA within")
P = {int(r.share_of_burden): r for r in PAR.itertuples()}
T155 = TOPK[TOPK.top_k == 155].iloc[0]
A = {int(r.archetype): r for r in ARCH.itertuples()}
a12_m1 = A[1].m1_children + A[2].m1_children
a12_m2 = A[1].m2_children + A[2].m2_children
a12_s1, a12_s2 = A[1].m1_share + A[2].m1_share, A[1].m2_share + A[2].m2_share
anc_r1, del_r1 = CAPS["_anc_r_m1"], CAPS["_deliv_r_m1"]
MC = {d["archetype"]: d for d in CAPS["_mc"]}

# ------------------------------------------------------------------ helpers
In = Inches


def rgb(h):
    return RGBColor.from_string(h)


def shape(slide, name):
    hits = [s for s in slide.shapes if s.name == name]
    assert len(hits) == 1, (name, len(hits))
    return hits[0]


def set_text(sh, paras):
    """Replace text keeping the shape's run formatting. paras: str | [runs] | [[runs], ...]; run = str | (str, bold)."""
    if isinstance(paras, str):
        paras = [[paras]]
    elif paras and not isinstance(paras[0], list):
        paras = [paras]
    txb = sh.text_frame._txBody
    ps = txb.findall(qn("a:p"))
    runs = [r for p in ps for r in p.findall(qn("a:r"))]
    bold_t = next((r for r in runs if r.find(qn("a:rPr")) is not None and r.find(qn("a:rPr")).get("b") == "1"), None)
    reg_t = next((r for r in runs if r.find(qn("a:rPr")) is None or r.find(qn("a:rPr")).get("b") != "1"), None)
    first = runs[0]
    p0 = copy.deepcopy(ps[0])
    for ch in list(p0):
        if ch.tag in (qn("a:r"), qn("a:br"), qn("a:fld")):
            p0.remove(ch)
    for p in ps:
        txb.remove(p)
    for spec in paras:
        p = copy.deepcopy(p0)
        end = p.find(qn("a:endParaRPr"))
        for run in spec:
            txt, b = (run, None) if isinstance(run, str) else run
            tmpl = first if b is None else (bold_t if b else reg_t)
            if tmpl is None:
                tmpl = first
            r = copy.deepcopy(tmpl)
            r.find(qn("a:t")).text = txt
            rpr = r.find(qn("a:rPr"))
            if b is not None and rpr is not None:
                rpr.set("b", "1" if b else "0")
            if end is not None:
                end.addprevious(r)
            else:
                p.append(r)
        txb.append(p)


def sub_text(sh, old, new):
    """Substitute within runs; asserts the old text was found."""
    hit = False
    for p in sh.text_frame.paragraphs:
        for r in p.runs:
            if old in r.text:
                r.text = r.text.replace(old, new)
                hit = True
    assert hit, (sh.name, old)


def fit(png, w, h):
    iw, ih = Image.open(png).size
    ar = iw / ih
    return (w, w / ar) if w / ar <= h else (h * ar, h)


def replace_picture(slide, old, png, box=None, valign="middle"):
    x, y, w, h = box or (old.left / 914400, old.top / 914400, old.width / 914400, old.height / 914400)
    fw, fh = fit(png, w, h)
    px = x + (w - fw) / 2
    py = y + ((h - fh) / 2 if valign == "middle" else 0)
    pic = slide.shapes.add_picture(str(png), In(px), In(py), In(fw), In(fh))
    old._element.addprevious(pic._element)
    old._element.getparent().remove(old._element)
    return pic


def picture(slide, png, x, y, w, h, valign="top", halign="center"):
    fw, fh = fit(png, w, h)
    px = x + ((w - fw) / 2 if halign == "center" else 0)
    py = y + ((h - fh) / 2 if valign == "middle" else 0)
    return slide.shapes.add_picture(str(png), In(px), In(py), In(fw), In(fh))


def rect(slide, x, y, w, h, fill=LIGHT, line=None, shape_type=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape_type, In(x), In(y), In(w), In(h))
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, paras, size=14, color=INK, bold=False, italic=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=None):
    """paras: str | list of paragraphs; paragraph = str | list of runs; run = str | (str, dict(bold,color,size,italic))."""
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
        if spacing:
            p.space_after = Pt(spacing)
        for run in ([para] if isinstance(para, str) else para):
            t, o = (run, {}) if isinstance(run, str) else run
            r = p.add_run(); r.text = t
            f = r.font; f.name = "Arial"; f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold); f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return s


def header(slide, title, subtitle=None):
    text(slide, 0.60, 0.26, 12.13, 0.57, title, size=30, color=NAVY, bold=True)
    if subtitle:
        text(slide, 0.60, 0.87, 12.13, 0.34, subtitle, size=17, color=SUB)


def footer(slide, s):
    text(slide, 0.60, 6.80, 12.13, 0.45, s, size=11, color=FOOT, italic=True)


def kpi(slide, x, y, w, h, big, label, big_color=GREEN, fill=GTINT, line=GLINE, big_size=34, label_size=14):
    rect(slide, x, y, w, h, fill, line)
    text(slide, x + 0.2, y + 0.12, w - 0.4, 0.62, big, size=big_size, color=big_color, bold=True)
    text(slide, x + 0.2, y + 0.12 + big_size / 72 * 1.25, w - 0.4, h - 0.3 - big_size / 72 * 1.25, label,
         size=label_size, color=NAVY, bold=True)


def card(slide, x, y, w, h, lead, body, size=13, fill=LIGHT, line=LINE):
    rect(slide, x, y, w, h, fill, line)
    text(slide, x + 0.16, y + 0.1, w - 0.32, h - 0.2, [[(lead + " ", {"bold": True, "color": NAVY}), body]], size=size)


def table(slide, x, y, w, data, col_w, row_h, font=12, header_font=None, bold_rows=(), aligns=None,
          header_fill=NAVY, row_fills=None, colors=None, bold_cols=()):
    rows, cols = len(data), len(data[0])
    gf = slide.shapes.add_table(rows, cols, In(x), In(y), In(w), In(sum(row_h) if isinstance(row_h, list) else row_h * rows))
    tbl = gf.table
    tblPr = gf._element.graphic.graphicData.tbl.tblPr
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    tblPr.set("firstRow", "1"); tblPr.set("bandRow", "0")
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = In(cw)
    for i in range(rows):
        tbl.rows[i].height = In(row_h[i] if isinstance(row_h, list) else row_h)
        for j in range(cols):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = In(0.07); c.margin_top = c.margin_bottom = In(0.025)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            hdr = i == 0
            if hdr:
                c.fill.fore_color.rgb = rgb(header_fill)
            else:
                rf = (row_fills or {}).get(i)
                c.fill.fore_color.rgb = rgb(rf if rf else (LIGHT if i % 2 == 0 else WHITE))
            tf = c.text_frame; tf.word_wrap = True
            lines = str(data[i][j]).split("\n")
            for li, ln in enumerate(lines):
                p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
                p.alignment = (aligns[j] if aligns else (PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT)) if not hdr else \
                    (PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
                r = p.add_run(); r.text = ln
                f = r.font; f.name = "Arial"
                f.size = Pt((header_font or font) if hdr else (font if li == 0 else font - 2))
                f.bold = hdr or ((i in bold_rows or j in bold_cols) and li == 0)
                col = WHITE if hdr else ((colors or {}).get((i, j)) or (colors or {}).get(j) or (INK if li == 0 else FOOT))
                f.color.rgb = rgb(col)
    return gf


def delete_shape(sh):
    sh._element.getparent().remove(sh._element)


def new_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    for ph in list(s.placeholders):
        delete_shape(ph)
    return s


# ------------------------------------------------------------------ build
prs = Presentation(str(SRC))
S = prs.slides
n_orig = len(S)
assert n_orig == 33

# Slide 2: Executive Director requests (cards 2 and 3)
s = S[1]
set_text(shape(s, "TextBox 19"), "All 774 LGAs clustered on 15 social and structural measures, using no vaccination data. Five archetypes, "
         f"each with a matched response package. Two types hold {a12_m1 / 1e6:.2f} million missed children.")
set_text(shape(s, "TextBox 28"), f"All {m1['lgas']} modelled LGAs ranked by burden, each carrying its rate, cumulative share, priority band, "
         f"equity tier and archetype. {m1['pareto']['50']} LGAs cover half of the burden; {m1['pareto']['80']} cover 80 percent.")

# Slide 3: outline
set_text(shape(S[2], "Text 16"), "Modelling approach: two Bayesian methods, from state estimates to every LGA")

# Slide 7: methods at a glance (Domain 5 line)
s = S[6]
hits = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.startswith("Estimated where missed children are")]
assert len(hits) == 1
set_text(hits[0], "Estimated where missed children are (two methods) and grouped areas by barrier.")
hits = [sh for sh in s.shapes if sh.has_text_frame and "maximum R-hat = 1.00" in sh.text_frame.text]
assert len(hits) == 1
sub_text(hits[0], "maximum R-hat = 1.00", "maximum R-hat below 1.01")

# Slide 9: LGA coverage (rebuilt)
s = S[8]
keep = {"Text 0"}
for sh in list(s.shapes):
    if sh.name not in keep:
        delete_shape(sh)
set_text(shape(s, "Text 0"), "Local government coverage of the zero-dose estimates")
text(s, 0.60, 0.87, 12.13, 0.40, f"Method 1 estimates {m1['lgas']} of the 774 local government areas; Method 2 estimates all 774.",
     size=17, color=SUB)
zrows = [["Zone", "LGAs", "Method 1", "Method 2"]]
for r in ZN.itertuples():
    if r.area != "National":
        zrows.append([r.area, k(r.lgas_total), k(r.m1_lgas), k(r.m2_lgas)])
zrows.append(["Nigeria", k(nat.lgas_total), k(nat.m1_lgas), k(nat.m2_lgas)])
table(s, 0.60, 1.45, 6.0, zrows, [2.4, 1.2, 1.2, 1.2], 0.46, font=14, header_font=13, bold_rows=(7,),
      row_fills={7: "DCE6F0"})
kpi(s, 7.00, 1.45, 2.75, 1.55, f"{m1['lgas']} / 774", "LGAs estimated by Method 1", big_color=M1C, big_size=30, label_size=13)
kpi(s, 9.98, 1.45, 2.75, 1.55, "774 / 774", "LGAs estimated by Method 2 (SAE)", big_color=M2C, big_size=30, label_size=13)
card(s, 7.00, 3.20, 5.73, 2.05, "Guzamala (Borno).",
     "No Penta1 doses were reported in DHIS2 for 2021 to 2024, and only partially in 2025. Method 1 distributes each state's children using the LGA's share "
     "of reported Penta1 doses, so it cannot place Guzamala. Method 2 estimates it from the Borno survey anchor, local "
     "conditions and neighbouring LGAs.", size=13)
card(s, 0.60, 5.45, 12.13, 1.10, "What the data team can do.",
     "Restore routine Penta1 reporting for Guzamala, and review LGAs whose reported doses are far above or below their "
     "estimated child population in the monthly data-quality checks; these are the LGAs where the two methods differ most.",
     size=13, fill=GTINT, line=GLINE)
footer(s, "Source: DHIS2 routine antigen export by local government area and month, 2021 to 2025; GRID3 boundaries (774 LGAs). "
          "LGA = local government area; SAE = small-area estimation.")

# Slides 10-12: Domain 1 (antigen early-warning, corrected DHIS2 counts, 773 LGAs)
D1 = ROOT.parent / "Domain1_Update_20261009"
d1s = json.loads((D1 / "outputs" / "corrected" / "d1_summary.json").read_text())
d1w = pd.read_excel(D1 / "government_workbook" / "NPHCDA_Domain1_Antigen_EarlyWarning_LGA_2026.xlsx", sheet_name="Worklist (flagged)", header=2)
d1_conf = int(d1w["Decline already visible in 2025"].str.startswith("Yes").sum())
d1a = pd.read_csv(D1 / "outputs" / "corrected" / "D1_additional_antigen_forecast_summary.csv").set_index("Antigen")
rota25 = d1a.loc[["Rota1", "Rota2", "Rota3"], "2025 observed % of 2024"]
s = S[9]
replace_picture(s, shape(s, "Image 0"), D1 / "figures" / "D1_F4_lga_flag_map.png")
set_text(shape(s, "Text 3"), f"{d1s['alerts']:,}")
set_text(shape(s, "Text 4"), f"early-warning flags, in {d1s['lgas_with_any_alert']} of the {d1s['lgas_with_any_forecast']} LGAs")
set_text(shape(s, "Text 5"), f"{d1s['forecasts_fitted']:,} forecasts, one per LGA and tracer vaccine. For {d1_conf} flags the fall is "
         "already visible in 2025. Nationally, all four stay above their 2024 level.")
set_text(shape(s, "Text 8"), [("What to do. ", True), ("Follow up first where the fall is already visible, then watch the projected dips, by vaccine and by LGA.", False)])
set_text(shape(s, "Text 9"), "Model estimate. The 80 percent mark compares each LGA with its own 2024 level of doses delivered. It is a decline "
         "tripwire, not coverage of the eligible child population. Map: number of the four tracer vaccines flagged in each LGA. Source: Domain 1 "
         "Prophet forecasts on DHIS2 routine data, 2021 to 2025 (Guzamala, Borno, has no 2024 data and no forecast).")
s = S[10]
replace_picture(s, shape(s, "Image 0"), D1 / "figures" / "D1_F5_additional_antigens.png")
set_text(shape(s, "Text 0"), "No decline warning for the nine additional antigens")
set_text(shape(s, "Text 1"), "Same method, nine more vaccines. The established five hold; rotavirus uptake dipped in 2025 and needs watching.")
set_text(shape(s, "Text 7"), f"new series - IPV2 and Rotavirus 1 to 3 - are tracked for uptake. IPV2 is rising; rotavirus was "
         f"{rota25.min():.0f}-{rota25.max():.0f} percent of 2024 in 2025.")
sub_text(shape(s, "Text 11"), "projected to 2027", "projected to 2028")
s = S[11]
replace_picture(s, shape(s, "Image 0"), D1 / "figures" / "D1_F6_opv3_worked_example.png")

# Slide 17: Domain 5 overview (Method 1)
s = S[16]
replace_picture(s, shape(s, "Image 0"), FIG / "M1_02_state_hotspots_2026_2028.png")
set_text(shape(s, "Text 5"), f"{m1['national'] / 1e6:.2f} million")
set_text(shape(s, "Text 6"), f"zero-dose children in 2026 (95% interval {m1['national_lo95'] / 1e6:.2f} to "
         f"{m1['national_hi95'] / 1e6:.2f} million), across {m1['lgas']} modelled LGAs")
set_text(shape(s, "Text 8"), f"{m1['top155_share']:.0f}%")
set_text(shape(s, "Text 9"), "of the burden is in the top 20% of LGAs (155 LGAs)")
set_text(shape(s, "Text 10"), f"{m1['pareto']['80']} local government areas hold 80 percent - concentrated, but not the textbook 80/20.")
ft = shape(s, "Text 11")
set_text(ft, f"Model estimate. Source: {M1N} (Bayesian hierarchical Beta regression on NDHS 2008 to 2024 with a DHIS2 trend term; "
         "LGA allocation by DHIS2 Penta1 share and NPC 2022 population); GRID3 boundaries; Getis-Ord Gi* state hotspots (Queen contiguity).")

# Slide 18: ranked LGAs (Method 1)
s = S[17]
replace_picture(s, shape(s, "Image 0"), DFIG / "D_M1_lga_hotspots.png", valign="top")
set_text(shape(s, "Text 1"), f"The top {m1['pareto']['50']} reach half of zero-dose children; the top {m1['pareto']['80']} reach 80 percent.")
top = L[L.m1_rank.notna()].sort_values("m1_rank").head(10)
row_y = [1.70, 2.10, 2.50, 2.90, 3.30, 3.70, 4.10, 4.50, 4.90, 5.30]
# map row cells by y position
cells = {}
for sh in s.shapes:
    if sh.name.startswith("Text ") and sh.has_text_frame:
        y = round(sh.top / 914400, 2)
        if y in row_y:
            cells.setdefault(y, []).append(sh)
for y, r in zip(row_y, top.itertuples()):
    cs = sorted(cells[y], key=lambda q: q.left)
    assert len(cs) == 6, y
    vals = [str(int(r.m1_rank)), r.state, r.lga_clean, r.zone, k(r.m1_children), f"{r.m1_rate:.1f}"]
    for c, val in zip(cs, vals):
        set_text(c, val)
        if len(val) > 11:
            for r_ in c.text_frame.paragraphs[0].runs:
                r_.font.size = Pt(14)
set_text(shape(s, "Text 75"), "Top 10 shown here (Method 1). The full ranked list for both methods is in the government workbook "
         "and can be filtered by state or priority level in the web app.")
set_text(shape(s, "Text 76"), f"Model estimate. Source: {M1N}, 2026; GRID3 boundaries. Hotspot method: Getis-Ord Gi* "
         "(spatial clustering), k = 5 nearest neighbours. Rates are capped at 99%.")
set_text(shape(s, "Text 2"), "Red marks clusters of high zero-dose burden; blue marks low-burden clusters.")

# Slide 19: archetype map
s = S[18]
replace_picture(s, shape(s, "Image 0"), FIG / "A_01_archetype_map_burden.png")
set_text(shape(s, "Text 1"), [[("Built from social and structural data alone - education, nutrition, poverty, wealth, travel time, antenatal "
                                "care and conflict - the archetypes used no vaccination data. Yet they reproduce where the children are: "
                                f"Archetypes 1 and 2 hold {a12_m1 / 1e6:.2f} million of {m1['national'] / 1e6:.2f} million zero-dose "
                                f"children ({a12_s1:.0f} percent; {a12_s2:.0f} percent under Method 2).", None)]])
set_text(shape(s, "Text 2"), "Colour shows the type of problem. Bars compare each archetype's share of children with its share of "
         "zero-dose children under both methods. This helps teams decide both where to go and what to do.")
sub_text(shape(s, "Text 3"), "bubbles are modelled zero-dose children", "bars are modelled zero-dose children under both methods")

# Slide 20: archetype table
s = S[19]
set_text(shape(s, "Text 1"), f"Archetypes 1 and 2 hold about {a12_s1:.0f} percent of all zero-dose children "
         f"({a12_s2:.0f} percent under Method 2).")
for a_id, (lg, ch, rt) in {1: ("Text 13", "Text 14", "Text 15"), 2: ("Text 22", "Text 23", "Text 24"),
                           3: ("Text 30", "Text 31", "Text 32"), 4: ("Text 39", "Text 40", "Text 41"),
                           5: ("Text 47", "Text 48", "Text 49")}.items():
    a = A[a_id]
    set_text(shape(s, lg), k(a.lgas))
    set_text(shape(s, ch), [[k(a.m1_children)], [f"({a.m1_share:.0f}%)"]])
    set_text(shape(s, rt), f"{a.m1_mean_rate:.1f}%")
sh = shape(s, "Text 52")
set_text(sh, "Model estimate. Agglomerative (Ward) clustering of 15 covariates across all 774 local government areas. Zero-dose "
         f"children and mean LGA rate from Method 1 (total {m1['national'] / 1e6:.2f} million); Method 2 totals by archetype are in the "
         "government workbook. Grey text names example states. BHCPF = Basic Health Care Provision Fund.")

# Slide 21: maternal care
s = S[20]
replace_picture(s, shape(s, "Image 0"), FIG / "A_02_maternal_care.png")
set_text(shape(s, "Text 3"), f"{a12_m1 / 1e6:.1f} million")
set_text(shape(s, "Text 4"), f"of the {m1['national'] / 1e6:.2f} million modelled zero-dose children (Method 1) are in the two "
         "archetypes with the weakest maternal care.")
set_text(shape(s, "Text 6"), f"r = {anc_r1:.2f}")
set_text(shape(s, "Text 7"), "antenatal care, four or more visits, against the Method 1 zero-dose rate across local governments. "
         f"Facility delivery is r = {del_r1:.2f}.")
set_text(shape(s, "Text 9"), [("The link. ", True), ("Weaker antenatal care goes with higher zero-dose across the local "
                              f"governments with antenatal-care data (ANC4+ r = {anc_r1:.2f}; facility delivery r = {del_r1:.2f}). "
                              "The link is strong and consistent.", False)])
set_text(shape(s, "Text 11"), [("The burden. ", True), (f"Remote Rural and Conflict-Affected hold about {a12_m1 / 1e6:.1f} million "
                               "zero-dose children, on the lowest antenatal care (36 to 40%) and facility delivery (10 to 19%).", False)])

sub_text(shape(s, "Text 14"), " Bubble size = modelled zero-dose children.", "")

# Slide 22: RMNCH-EPI list
s = S[21]
for a_id, (zr, ch) in {1: ("Text 15", "Text 16"), 2: ("Text 22", "Text 23"), 3: ("Text 30", "Text 31"),
                       4: ("Text 37", "Text 38"), 5: ("Text 45", "Text 46")}.items():
    set_text(shape(s, zr), f"{A[a_id].m1_mean_rate:.1f}")
    set_text(shape(s, ch), k(A[a_id].m1_children))
sub_text(shape(s, "Text 50"), "modelled zero", "Method 1 modelled zero")

# Slide 25: NmDHS validation (Method 1)
s = S[24]
replace_picture(s, shape(s, "Image 0"), FIG / "M1_04_nmdhs_validation.png", box=(0.55, 1.55, 7.25, 3.7))
set_text(shape(s, "Text 3"), f"rho = {v1n.rho:.2f}")
nw = ZV[ZV.area == "North West"].iloc[0]
set_text(shape(s, "Text 6"), f"{nw.m1_rate:.0f}% vs {nw.nmdhs_2025_26:.0f}%")

# Slide 26: IHME concordance (Method 1)
s = S[25]
replace_picture(s, shape(s, "Image 0"), FIG / "M1_05_ihme_concordance.png")
set_text(shape(s, "Text 3"), f"rho = {v1i.rho:.2f}")
set_text(shape(s, "Text 5"), "Most LGAs were placed in the same or a neighbouring priority third. Only "
         f"{CAPS['_m1_ihme_opposite']} were in opposite thirds and should be reviewed.")
sub_text(shape(s, "Text 7"), "730 matched areas", f"{int(v1i.n)} matched LGAs")
sub_text(shape(s, "Text 7"), "our Bayesian local-government estimate", "Method 1 local-government estimate")

# Slide 27: how the RI team uses this
s = S[26]
replace_picture(s, shape(s, "Image 0"), FIG / "M1_03_pareto.png")
set_text(shape(s, "Text 20"), [("Sequencing the money. ", True),
                               (f"Start with about {m1['pareto']['50']} LGAs to reach half of missed children, expand to about "
                                f"{m1['pareto']['60']} for 60 percent, then toward {m1['pareto']['80']} LGAs to reach 80 percent.", False)])
set_text(shape(s, "Text 21"), f"Model estimate. Source: Domain 5 local-government ranking and Pareto curve ({M1N}; {m1['lgas']} "
         f"local government areas; national total {m1['national'] / 1e6:.2f} million).")

# Slide 31: recommendations
s = S[30]
set_text(shape(s, "Text 5"), f"Begin with Sokoto, Kebbi, Zamfara and the highest-burden LGAs. About {m1['pareto']['50']} LGAs reach half "
         f"of missed children; the {R['overlap_top155']} LGAs ranked in the top 155 by both methods come first.")

# Slide 33: acronyms (add a row)
s = S[32]
rect(s, 0.60, 6.34, 5.85, 0.50, LIGHT)
rect(s, 6.88, 6.34, 5.85, 0.50, LIGHT)
for x0, a_, b_ in [(0.74, "SAE", "Small-area estimation"), (7.02, "BYM2", "Besag-York-Mollie spatial model, version 2")]:
    text(s, x0, 6.34, 1.75, 0.50, a_, size=14, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x0 + 1.79, 6.34, 3.78, 0.50, b_, size=14, color=INK, anchor=MSO_ANCHOR.MIDDLE)
sh = shape(s, "Text 52"); sh.top = In(6.98)

# ================================================================== new slides
NEW = []

# N1 Method 2 overview
s = new_slide(prs); NEW.append(s)
header(s, "Method 2: Bayesian small-area estimation (SAE)",
       "A second, survey-anchored model that estimates zero-dose directly for every one of the 774 LGAs.")
picture(s, FIG / "M2_01_model_schematic.png", 0.60, 1.40, 7.35, 3.75)
steps = [("Anchored to the surveys.", "Each state's LGA estimates must add up to that state's NDHS zero-dose level (2008, 2013, 2018, "
          "2023-24), allowing for survey sampling error."),
         ("Local conditions.", "Six LGA measures shape differences within a state: maternal care, improved water, wealth, travel "
          "time to a facility, conflict and poverty."),
         ("Neighbours.", "A spatial term (BYM2) lets neighbouring LGAs share information, so sparse areas borrow strength from "
          "their surroundings.")]
for i, (lead, body) in enumerate(steps):
    y = 1.40 + i * 1.27
    rect(s, 8.20, y, 0.42, 0.42, NAVY, shape_type=MSO_SHAPE.OVAL)
    text(s, 8.20, y, 0.42, 0.42, str(i + 1), size=15, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.78, y - 0.02, 3.95, 1.2, [[(lead + " ", {"bold": True, "color": NAVY}), body]], size=13)
card(s, 0.60, 5.35, 12.13, 1.25, "Why a second method.",
     "Method 1 distributes each state's estimate to its LGAs using routine DHIS2 Penta1 reporting. Method 2 does not use routine "
     "coverage at all: it combines survey levels with local conditions and geography, and gives every LGA its own 95% credible "
     "interval and a probability of being a priority. Reading both together shows which priorities are robust.",
     size=13, fill=GTINT, line=GLINE)
footer(s, "Model estimate. NDHS = Nigeria Demographic and Health Survey; BYM2 = Besag-York-Mollie spatial model, version 2. Fitted in PyMC "
          "(4 chains x 8,000 draws). Covariates: DHS spatial surfaces 2018, Meta Relative Wealth Index, travel time to healthcare, ACLED, poverty.")

# N2 rate + uncertainty
s = new_slide(prs); NEW.append(s)
header(s, "Method 2: zero-dose rate and its uncertainty, all 774 LGAs",
       "Every LGA has its own estimate and its own 95% credible interval.")
picture(s, FIG / "M2_02_rate_and_uncertainty_maps.png", 0.60, 1.35, 8.75, 3.95)
nr2 = nat.m2_rate
kpi(s, 9.65, 1.35, 3.08, 1.85, f"{nr2:.1f}%",
    f"national zero-dose rate, 2026 (95% CrI {nat.m2_lo95 / nat.cohort_12_23m * 100:.1f} to {nat.m2_hi95 / nat.cohort_12_23m * 100:.1f}%)",
    big_color=M2C, big_size=32, label_size=12)
nwz = ZN[ZN.area == "North West"].iloc[0]
kpi(s, 9.65, 3.40, 3.08, 1.85, f"{nwz.m2_rate:.1f}%", "North-West zone rate, the highest of the six zones",
    big_color=M2C, big_size=32, label_size=12, fill=LIGHT, line=LINE)
card(s, 0.60, 5.45, 12.13, 1.15, "How to read the right-hand map.",
     "Wider intervals mark LGAs where survey anchors and local measures leave more room for error. These LGAs are the first "
     "candidates for local verification, for example a rapid coverage assessment, before large investments.", size=13)
footer(s, f"Model estimate. {M2N}, 2026. CrI = credible interval. Sources: NDHS 2008-2024; NPC 2022; GRID3 boundaries; covariates as listed on the method slide.")

# N3 burden + priority probability
s = new_slide(prs); NEW.append(s)
header(s, "Method 2: zero-dose burden and priority probability",
       "Burden combines rate and child numbers; the probability shows how stable each priority is.")
picture(s, FIG / "M2_03_burden_and_priority_maps.png", 0.60, 1.35, 8.75, 3.95)
kpi(s, 9.65, 1.35, 3.08, 1.85, f"{m2['national'] / 1e6:.2f} million",
    f"zero-dose children in 2026 (95% CrI {m2['national_lo95'] / 1e6:.2f} to {m2['national_hi95'] / 1e6:.2f} million)",
    big_color=M2C, big_size=28, label_size=12)
kpi(s, 9.65, 3.40, 3.08, 1.85, f"{m2['top155_share']:.0f}%", "of the burden is in the top 20% of LGAs (155 LGAs)",
    big_color=M2C, big_size=32, label_size=12, fill=LIGHT, line=LINE)
card(s, 0.60, 5.45, 12.13, 1.15, "Priority probability.",
     f"For each LGA, the share of 4,000 model simulations in which it ranks among the 155 highest-burden LGAs. "
     f"{m2['near_certain_top155']} LGAs are priorities in at least 90% of simulations; LGAs between 0.1 and 0.9 are borderline and "
     "should be confirmed with local data.", size=13)
footer(s, f"Model estimate. {M2N}, 2026. Children aged 12-23 months from the 2024 under-five projection divided by five, shared by NPC 2022 LGA population.")

# N4 top 20
top20 = L.nsmallest(20, "m2_rank")
s = new_slide(prs); NEW.append(s)
header(s, "Method 2: the 20 highest-burden LGAs",
       "Each bar shows the 95% credible interval; P is the probability of being among the top 155.")
picture(s, FIG / "M2_04_top20_lgas.png", 0.60, 1.35, 7.6, 5.3, halign="left")
zc = top20.zone.value_counts()
sc = top20.state.value_counts()
kpi(s, 8.55, 1.35, 4.18, 1.6, f"{zc.get('North West', 0)} of 20", "are in the North-West zone",
    big_color=M2C, big_size=32, label_size=13)
kpi(s, 8.55, 3.10, 4.18, 1.6, f"{int((top20.m2_p_top155 >= 0.9).sum())} of 20", "are priorities in at least 90% of simulations",
    big_color=M2C, big_size=32, label_size=13, fill=LIGHT, line=LINE)
card(s, 8.55, 4.85, 4.18, 1.75, "Concentration.",
     ", ".join(f"{st} {n}" for st, n in sc.items()) + ". These LGAs combine high rates with large child populations.", size=12)
footer(s, f"Model estimate. {M2N}, 2026. Ranks from 4,000 posterior draws.")

# N5 diagnostics
s = new_slide(prs); NEW.append(s)
header(s, "Method 2: model performance and diagnostics",
       "The model fits every survey round, passes an independent survey check, and its sampler converged cleanly.")
picture(s, FIG / "M2_05_diagnostics.png", 0.60, 1.30, 12.13, 4.15)
d2 = m2["diagnostics"]
tiles = [(f"{d2['max_rhat']:.3f}", "maximum R-hat (target below 1.01)"),
         (k(d2["min_ess_bulk"]), "minimum effective sample size (target above 400)"),
         (str(d2["divergences"]), "divergent transitions"),
         (f"{m2['survey_ppc_coverage95'] * 100:.0f}%", "of NDHS state-round values inside the 95% predictive interval"),
         (f"rho {v2n.rho:.2f}", f"against NmDHS 2025-26 (37 states); mean absolute error {v2n.MAE_pp:.1f} points")]
tw = (12.13 - 4 * 0.15) / 5
for i, (big, lab) in enumerate(tiles):
    x = 0.60 + i * (tw + 0.15)
    rect(s, x, 5.55, tw, 1.12, GTINT, GLINE)
    text(s, x + 0.12, 5.6, tw - 0.24, 0.42, big, size=20, color=GREEN, bold=True)
    text(s, x + 0.12, 6.02, tw - 0.24, 0.62, lab, size=10.5, color=NAVY, bold=True)
footer(s, f"{M2N}. {d2['parameters']:,} parameters; 4 chains x 8,000 draws after 3,000 tuning steps. NmDHS = Nigeria mini Demographic "
          "and Health Survey, not used in fitting.")

# N6 covariates
s = new_slide(prs); NEW.append(s)
header(s, "Method 2: what explains differences between LGAs",
       "Maternal care is the strongest and most certain predictor of zero-dose within states.")
picture(s, FIG / "M2_06_covariate_effects.png", 0.60, 1.35, 7.0, 4.0, halign="left")
import math  # noqa: E402
crow = [["Local measure", "Effect (95% CrI)", "Odds ratio"]]
for r in COV.itertuples():
    crow.append([r.covariate.replace(" (ANC4+, facility delivery)", ""), f"{r.mean:+.2f} ({r.lo95:+.2f} to {r.hi95:+.2f})",
                 f"{math.exp(r.mean):.2f}"])
table(s, 7.85, 1.35, 4.88, crow, [1.95, 1.95, 0.98], 0.45, font=11, header_font=11)
card(s, 7.85, 4.75, 4.88, 1.85, "Reading the table.",
     "Effects are on the log-odds scale per standard deviation; below zero means lower zero-dose. An odds ratio of "
     f"{math.exp(COV.iloc[0]['mean']):.2f} means one standard deviation better maternal care is linked with about "
     f"{(1 - math.exp(COV.iloc[0]['mean'])) * 100:.0f}% lower odds of a child being zero-dose.", size=12)
card(s, 0.60, 5.55, 7.0, 1.05, "Associations, not causes.",
     "These are partial associations used to distribute survey levels within states, not causal effects.", size=12)
footer(s, f"Model estimate. {M2N}. Maternal care = mean of standardised ANC4+ and facility delivery (DHS spatial surfaces, 2018).")

# N7 admin vs survey
s = new_slide(prs); NEW.append(s)
header(s, "Routine coverage and survey-anchored estimates",
       "Why Method 2 uses survey levels rather than routine coverage to estimate LGA zero-dose.")
picture(s, FIG / "M2_07_admin_vs_survey.png", 0.60, 1.35, 8.95, 3.6, halign="left")
kpi(s, 9.75, 1.35, 2.98, 1.12, f"{CAPS['_admin_gt100'] * 100:.0f}%", "of LGAs show Penta1 coverage above 100%",
    big_color="C0392B", fill=LIGHT, line=LINE, big_size=26, label_size=11)
kpi(s, 9.75, 2.60, 2.98, 1.12, f"rho {CAPS['_admin_rho']:.2f}", "routine-only zero-dose vs NmDHS 2025-26, 37 states",
    big_color="C0392B", fill=LIGHT, line=LINE, big_size=26, label_size=11)
kpi(s, 9.75, 3.85, 2.98, 1.12, f"rho {v2n.rho:.2f}", "Method 2 vs NmDHS 2025-26, 37 states",
    big_color=M2C, big_size=26, label_size=11)
card(s, 0.60, 5.15, 12.13, 1.45, "What this means.",
     "Administrative coverage divides doses given by an estimated target population. When the target population is too small, "
     "or doses include children from other LGAs, coverage exceeds 100 percent and cannot be read as a rate. Routine data remain "
     "essential for monitoring trends and data quality; for the level of zero-dose in each LGA, survey-anchored estimates are "
     f"more reliable (routine-only state estimates differ from NmDHS by {CAPS['_admin_mae']:.0f} points on average).",
     size=12.5, fill=GTINT, line=GLINE)
footer(s, "DHIS2 Penta1 2023-2025 annual mean over the 2024 cohort projection (state under-five / 5, NPC 2022 shares). Sources: DHIS2; NmDHS 2025-26.")

# N8 zone comparison
s = new_slide(prs); NEW.append(s)
header(s, "Method 1 and Method 2 compared by zone",
       "About half of zero-dose children are in the North-West under both methods; national totals differ by 2.4%.")
zt = [["Zone", "LGAs\n(M1 / M2)", "Children\n12-23 months", "Method 1\nzero-dose children (95% interval)", "Method 1\nrate %",
       "Method 2\nzero-dose children (95% CrI)", "Method 2\nrate %", "Difference\nM2 vs M1", "NmDHS\n2025-26 %"]]
order = ["North West", "North East", "North Central", "South West", "South South", "South East", "National"]
for z in order:
    r = ZN[ZN.area == z].iloc[0]
    nm = ZV[ZV.area == z].nmdhs_2025_26
    zt.append([("Nigeria" if z == "National" else z), f"{int(r.m1_lgas)} / {int(r.m2_lgas)}", k(r.cohort_12_23m),
               f"{k(r.m1_children)}\n({k(r.m1_lo95)} - {k(r.m1_hi95)})", f"{r.m1_rate:.1f}",
               f"{k(r.m2_children)}\n({k(r.m2_lo95)} - {k(r.m2_hi95)})", f"{r.m2_rate:.1f}",
               f"{r.difference_pct:+.1f}%", (f"{nm.iloc[0]:.1f}" if len(nm) else "-")])
cw = [1.6, 0.95, 1.25, 2.2, 0.9, 2.2, 0.9, 1.05, 0.98]
table(s, 0.60, 1.32, sum(cw), zt, cw, [0.58] + [0.47] * 7, font=11, header_font=10, bold_rows=(7,), row_fills={7: "DCE6F0"},
      colors={3: M1C, 4: M1C, 5: M2C, 6: M2C})
s_nw1 = ZN[ZN.area == "North West"].iloc[0]
card(s, 0.60, 5.55, 3.93, 1.08, "Same national picture.",
     f"{m1['national'] / 1e6:.2f} million (Method 1) and {m2['national'] / 1e6:.2f} million (Method 2); the 95% intervals overlap "
     "in every zone.", size=12)
card(s, 4.70, 5.55, 3.93, 1.08, "North-West first.",
     f"{s_nw1.m1_share:.0f}% of zero-dose children under Method 1 and {s_nw1.m2_share:.0f}% under Method 2.", size=12)
ne = ZN[ZN.area == "North East"].iloc[0]
card(s, 8.80, 5.55, 3.93, 1.08, "Largest gap: North-East.",
     f"Method 2 is {ne.difference_pct:.0f}% higher ({ne.m2_rate:.1f}% vs {ne.m1_rate:.1f}%); NmDHS 2025-26 shows 38.7%.", size=12)
footer(s, f"Model estimate, 2026. {M1N} (773 LGAs); {M2N} (774 LGAs). NmDHS 2025-26 zone rates for reference (not used in fitting).")

# N9 state comparison
s = new_slide(prs); NEW.append(s)
from scipy.stats import spearmanr  # noqa: E402
rs = spearmanr(ST.m1_children, ST.m2_children).statistic
header(s, "Method 1 and Method 2 compared by state",
       f"The methods rank the 37 states almost identically (rho = {rs:.2f}); ordered by Method 1 burden.")
so = ST.sort_values("m1_children", ascending=False).reset_index(drop=True)
hdr = ["State", "Method 1\nchildren", "Method 2\nchildren", "M1\nrate %", "M2\nrate %", "NmDHS\n%"]
cw = [1.35, 1.05, 1.05, 0.75, 0.75, 0.9]
for half, (a0, a1) in enumerate([(0, 19), (19, 37)]):
    rows = [hdr] + [[r.state, k(r.m1_children), k(r.m2_children), f"{r.m1_rate:.1f}", f"{r.m2_rate:.1f}", f"{r.nmdhs_2025_26:.1f}"]
                    for r in so.iloc[a0:a1].itertuples()]
    table(s, 0.60 + half * 6.23, 1.32, sum(cw), rows, cw, [0.44] + [0.255] * (len(rows) - 1), font=10, header_font=9.5,
          colors={1: M1C, 3: M1C, 2: M2C, 4: M2C})
footer(s, f"Model estimate, 2026. {M1N}; {M2N}. Children = modelled zero-dose children aged 12-23 months. NmDHS 2025-26 for reference.")

# N10 national + independent agreement
s = new_slide(prs); NEW.append(s)
header(s, "National comparison and agreement with independent data",
       "Both methods match the new national survey; Method 2 agrees more closely with the independent IHME LGA map.")


def ci(r):
    return f"{r.rho:.2f} ({r.ci_lo:.2f} to {r.ci_hi:.2f})"


nt = [["Measure", "Method 1", "Method 2 (SAE)"],
      ["LGAs estimated", f"{m1['lgas']}", f"{m2['lgas']}"],
      ["Zero-dose children, 2026\n(95% interval)", f"{m1['national'] / 1e6:.2f} million\n({m1['national_lo95'] / 1e6:.2f} - {m1['national_hi95'] / 1e6:.2f})",
       f"{m2['national'] / 1e6:.2f} million\n({m2['national_lo95'] / 1e6:.2f} - {m2['national_hi95'] / 1e6:.2f})"],
      ["National zero-dose rate", f"{nat.m1_rate:.1f}%", f"{nat.m2_rate:.1f}%"],
      ["NmDHS 2025-26, 37 states:\nSpearman rho (95% CI)", ci(v1n), ci(v2n)],
      ["NmDHS 2025-26: mean absolute\nerror / bias, points", f"{v1n.MAE_pp:.1f} / {v1n.bias_pp:+.1f}", f"{v2n.MAE_pp:.1f} / {v2n.bias_pp:+.1f}"],
      [f"IHME DTP1 2018, {int(v1i.n)} LGAs:\nSpearman rho (95% CI)", ci(v1i), ci(v2i)],
      ["IHME within states:\nSpearman rho (95% CI)", ci(v1w), ci(v2w)],
      ["States with positive\nwithin-state agreement", f"{int(v1w.states_positive)} of {int(v1w.states_tested)}",
       f"{int(v2w.states_positive)} of {int(v2w.states_tested)}"]]
cw = [2.85, 2.05, 2.05]
table(s, 0.60, 1.35, sum(cw), nt, cw, [0.42] + [0.56] * 8, font=12, header_font=12, colors={1: M1C, 2: M2C},
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
picture(s, FIG / "C_03_agreement_independent_data.png", 7.80, 1.35, 4.93, 2.6)
card(s, 7.80, 4.15, 4.93, 2.45, "How to read this.",
     "Both methods rank states almost identically against NmDHS 2025-26, which neither model used. At LGA level, Method 2 "
     "agrees far more closely with the independent IHME map, above all within states, where Method 1 follows routine "
     f"reporting patterns. {R['overlap_top155']} of the top 155 LGAs are shared by both methods.", size=12, fill=GTINT, line=GLINE)
footer(s, "rho = Spearman rank correlation; 95% CI from 2,000 bootstrap resamples. All p < 0.001 except Method 1 within states (p = "
          f"{v1w.p_value:.2f}). IHME = Institute for Health Metrics and Evaluation Local Burden of Disease DTP1, 2018 (zero-dose = 100 - DTP1).")

# N11 Pareto both methods
s = new_slide(prs); NEW.append(s)
header(s, "LGAs needed to reach 50, 60 and 80 percent of the burden",
       "Burden concentration under both methods, ranking LGAs from highest to lowest modelled burden.")
picture(s, FIG / "C_04_pareto_both_methods.png", 0.60, 1.35, 7.35, 4.6, halign="left")
pt = [["Share of\nzero-dose children", "Method 1\nLGAs (% of LGAs)", "Method 2\nLGAs (% of LGAs)"]]
for sh_ in (50, 60, 80):
    r = P[sh_]
    pt.append([f"{sh_}%", f"{int(r.m1_lgas)} ({r.m1_pct_of_lgas:.0f}%)", f"{int(r.m2_lgas)} ({r.m2_pct_of_lgas:.0f}%)"])
pt.append(["Held by top 155\n(20% of LGAs)", f"{T155.m1_share:.1f}%\n({T155.m1_lo95:.1f} - {T155.m1_hi95:.1f})",
           f"{T155.m2_share:.1f}%\n({T155.m2_lo95:.1f} - {T155.m2_hi95:.1f})"])
cw = [1.62, 1.52, 1.52]
table(s, 8.10, 1.35, sum(cw), pt, cw, [0.62, 0.48, 0.48, 0.48, 0.66], font=13, header_font=10.5, colors={1: M1C, 2: M2C},
      aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER], bold_rows=(1, 2, 3))
card(s, 8.10, 4.35, 4.63, 2.25, "Planning rule of thumb.",
     f"About 140 to 150 LGAs reach half of all zero-dose children, about 180 to 200 reach 60 percent, and 320 to 340 reach "
     "80 percent, whichever method is used. Method 2 is slightly more concentrated.", size=12.5, fill=GTINT, line=GLINE)
footer(s, f"Model estimate, 2026. {M1N} (773 LGAs); {M2N} (774 LGAs). Method 2 band = 95% credible interval of the cumulative share.")

# N12 LGA agreement
s = new_slide(prs); NEW.append(s)
header(s, "Where the two methods agree and differ at LGA level",
       "Strong agreement overall; differences where routine reporting is out of line with population.")
picture(s, FIG / "C_05_lga_scatter.png", 0.60, 1.35, 8.3, 3.85, halign="left")
kpi(s, 9.15, 1.35, 3.58, 1.85, f"rho = {R['rank_corr']:.2f}", "rank agreement in zero-dose children across 773 LGAs",
    big_color=NAVY, fill=LIGHT, line=LINE, big_size=30, label_size=12)
kpi(s, 9.15, 3.35, 3.58, 1.85, f"{R['overlap_top155']} of 155", "top-20% LGAs shared by both methods: the most robust priorities",
    big_size=30, label_size=12)
card(s, 0.60, 5.35, 12.13, 1.25, "Where they differ.",
     "Method 1 follows each LGA's share of reported Penta1 doses: very high reported doses push its rate toward the 1% floor "
     "(for example Rabah and Goronyo, Sokoto) and low reported doses push it up (for example Sokoto North). Method 2 is not "
     "affected by reporting volumes. LGAs with large gaps are priorities for data-quality review.", size=12.5)
footer(s, f"Model estimate, 2026. {M1N}; {M2N}. Spearman rank correlation of modelled zero-dose children.")

# N13 which method
s = new_slide(prs); NEW.append(s)
header(s, "Which estimate for which decision",
       "The two methods answer the same question in different ways; each has a role.")
wt = [["Decision", "Recommended estimate", "Why"],
      ["National, zone and state totals", "Either; report Method 1 with Method 2 as the sensitivity range",
       f"Totals agree within {abs(nat.difference_pct):.1f}% nationally; state rank agreement rho = {rs:.2f}; both match NmDHS 2025-26"],
      ["Ranking LGAs within a state", "Method 2 (SAE)",
       f"Covers all 774 LGAs; full uncertainty for each LGA; within-state agreement with IHME rho = {v2w.rho:.2f} vs {v1w.rho:.2f}"],
      ["Selecting priority LGAs for funding", f"The {R['overlap_top155']} LGAs in the top 155 under both methods first",
       "Robust to the choice of method; then add LGAs with high Method 2 priority probability"],
      ["Tracking change and early warning", "Method 1",
       "Uses the DHIS2 trend and updates with monthly routine data"],
      ["Data-quality review", "LGAs where the methods differ most",
       "Large gaps flag reported doses out of line with population"]]
cw = [3.1, 3.9, 5.13]
table(s, 0.60, 1.40, sum(cw), wt, cw, [0.45] + [0.85] * 5, font=13, header_font=13, bold_cols=(0, 1),
      colors={1: NAVY}, aligns=[PP_ALIGN.LEFT] * 3)
footer(s, f"{M1N}; {M2N}. IHME = Institute for Health Metrics and Evaluation; NmDHS = Nigeria mini Demographic and Health Survey.")

# ---------------------------------------------------------------- reorder: new slides after original slide 22
lst = prs.slides._sldIdLst
ids = list(lst)
new_ids = ids[n_orig:]
for e in new_ids:
    lst.remove(e)
anchor = list(lst)[21]
for e in reversed(new_ids):
    anchor.addnext(e)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print("saved", OUT, len(prs.slides), "slides")
