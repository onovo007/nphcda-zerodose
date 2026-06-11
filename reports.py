"""
Generate premium deliverables from the live model outputs:
- a one-page factsheet (branded HTML, print-to-PDF),
- a policy brief (editable, branded Word .docx).

Findings are assembled from the same cached model functions the domain pages use, so values
match the on-screen results exactly. Narrative is LLM-drafted (grounded) when a key is present,
otherwise a templated narrative built from the numbers.
"""
from __future__ import annotations

import io
import re
from datetime import datetime

import pandas as pd

import config as C
import data_io as dio
import llm
from theme import clean, img_data_uri
from models.d1_forecast import national_forecasts
from models.d2_dropout import dropout_forecasts, lasso_drivers
from models.d5_zerodose import run_state_model, run_lga_burden


# --------------------------------------------------------------------------------------
# Findings assembly (reuses cached model results)
# --------------------------------------------------------------------------------------
def build_findings(data: dict) -> dict:
    f: dict = {"generated": datetime.now().strftime("%d %B %Y"),
               "consortium": "CIDRE and Quantium Insights LLC, in technical support of NPHCDA",
               "audience": "NPHCDA, GAVI and UNICEF"}
    kd = dio.df_hash(data.get("dhis2"))

    # Domain 5 (centerpiece)
    if all(data.get(k) is not None for k in ("ndhs_long", "under5", "dhis2", "lga_population")):
        kn, ku, kp = (dio.df_hash(data["ndhs_long"]), dio.df_hash(data["under5"]),
                      dio.df_hash(data["lga_population"]))
        mkey = f"{kn}-{ku}-{kd}-{C.MCMC_DRAWS_LIVE}-{C.MCMC_TUNE_LIVE}"
        out = run_state_model(data["ndhs_long"], data["under5"], data["dhis2"], key=mkey,
                              draws=C.MCMC_DRAWS_LIVE, tune=C.MCMC_TUNE_LIVE)
        res = out["res"]
        lga = run_lga_burden(data["dhis2"], res, data["lga_population"], key=f"{mkey}-{kp}")
        tier1 = res[res["priority_tier"].astype(str) == "Tier 1: Critical"]["state"].tolist()
        f["d5"] = {
            "national_zd_count_2026": int(res["zd_count_2026"].sum()),
            "lga_total": lga["national_total"], "lga_count": lga["n_lgas"],
            "top20_pct": lga["top20_pct"], "n80": lga["n80"],
            "tier1_states": tier1, "max_rhat": out["max_rhat"],
            "top_states": [
                {"state": r["state"], "zone": r["zone"],
                 "zd_2026_pct": round(float(r["zd_pred_2026_mean"]), 1),
                 "zd_2026_count": int(r["zd_count_2026"]) if pd.notna(r["zd_count_2026"]) else None,
                 "tier": str(r["priority_tier"])}
                for _, r in res.sort_values("state_rank").head(8).iterrows()],
            "top_lgas": [
                {"lga": r["LGA"], "state": r["State"], "zone": r["Zone"],
                 "zd_count": int(r["Zero-dose children (est)"]),
                 "zd_rate_pct": float(r["Zero-dose rate (%)"])}
                for _, r in lga["pareto"].head(10).iterrows()],
        }

    # Domain 1
    if data.get("dhis2") is not None:
        nat = dio.national_monthly(dio.prep_dhis2(data["dhis2"]))
        d1 = national_forecasts(nat, key=kd)
        sm = d1["summary"]
        f["d1"] = {
            "at_risk_antigens": sm[sm["Crosses 80% in 6-12m"] == "Yes"]["Antigen"].tolist(),
            "summary": sm.to_dict(orient="records"),
        }
        # Domain 2
        d2 = dropout_forecasts(nat, key=kd)
        f["d2"] = {"latest_dropout": {s["label"]: round(s["obs_y"][-1], 1) for s in d2.values()}}
        if data.get("model_dataset") is not None:
            agg = dio.state_monthly(dio.prep_dhis2(data["dhis2"]))
            drv = lasso_drivers(agg, data["model_dataset"])
            f["d2"]["top_drivers"] = {
                C.DROPOUT_TARGETS[t]: [k.replace("pct_", "").replace("_", " ") for k in c.head(4).index]
                for t, c in drv.items()}
    return f


# --------------------------------------------------------------------------------------
# Templated narrative (used when no LLM key)
# --------------------------------------------------------------------------------------
def template_narrative(f: dict, kind: str) -> str:
    d5 = f.get("d5", {})
    d1 = f.get("d1", {})
    d2 = f.get("d2", {})
    parts = []
    if kind == "policy":
        parts.append("## Executive summary")
    parts.append(
        f"Modelling of Nigeria routine immunization data projects about "
        f"{d5.get('lga_total', 0):,} zero-dose children across {d5.get('lga_count', 0)} LGAs in 2026. "
        f"Burden is concentrated: the top 20 percent of LGAs carry about {d5.get('top20_pct', 0):.0f} "
        f"percent of the total, and the highest-risk states are "
        f"{', '.join(d5.get('tier1_states', [])[:3]) or 'in the North-West'}.")
    if kind == "policy":
        parts.append("## Situation analysis")
        parts.append(
            "Zero-dose rates remain highest in the North-West. Antigen coverage is forecast near or "
            "above the 80 percent target nationally, but dropout between antigen doses and spatial "
            "clustering of unvaccinated children sustain the burden in specific LGAs.")
    parts.append("## Key findings" if kind == "policy" else "### Key findings")
    kf = []
    if d5:
        kf.append(f"About {d5.get('lga_total', 0):,} zero-dose children in 2026 across "
                  f"{d5.get('lga_count', 0)} LGAs; 80 percent of the burden sits in the top "
                  f"{d5.get('n80', 0)} LGAs.")
        kf.append("Tier-1 critical states: " + (", ".join(d5.get("tier1_states", [])) or "North-West states") + ".")
        if d5.get("top_lgas"):
            t = d5["top_lgas"][0]
            kf.append(f"Highest-burden LGA: {t['lga']} ({t['state']}), about {t['zd_count']:,} "
                      f"zero-dose children at {t['zd_rate_pct']:.0f} percent.")
    if d1:
        ar = d1.get("at_risk_antigens")
        kf.append(("Antigens at risk of falling below 80 percent in 6 to 12 months: " + ", ".join(ar))
                  if ar else "All tracer antigens are forecast at or above the 80 percent target nationally.")
    if d2.get("top_drivers"):
        first = next(iter(d2["top_drivers"].items()))
        kf.append(f"Leading drivers of {first[0]} dropout: {', '.join(first[1])}.")
    parts.append("\n".join(f"- {x}" for x in kf))
    parts.append("## Recommendations" if kind == "policy" else "### Priority actions")
    recs = [
        f"Prioritize the top {d5.get('n80', 0) if d5 else 'priority'} LGAs that hold 80 percent of the "
        "burden for targeted catch-up and outreach.",
        "Concentrate first-line investment in the Tier-1 states ("
        + (", ".join(d5.get("tier1_states", [])[:3]) if d5 else "North-West") + ").",
        "Address dropout with reminder-recall and defaulter tracing where dose-to-dose loss is highest.",
        "Use the LGA hotspot map to micro-plan supervision and supplementary sessions.",
    ]
    parts.append("\n".join(f"{i+1}. {r}" for i, r in enumerate(recs)))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------------------
def _strip_md(s: str) -> str:
    return clean(re.sub(r"\*\*(.*?)\*\*", r"\1", s).replace("**", "").strip())


def md_to_html(md: str) -> str:
    html, in_list = [], False
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("## "):
            if in_list:
                html.append("</ul>"); in_list = False
            html.append(f"<h2>{_strip_md(line[3:])}</h2>")
        elif line.startswith("### "):
            if in_list:
                html.append("</ul>"); in_list = False
            html.append(f"<h3>{_strip_md(line[4:])}</h3>")
        elif re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            if not in_list:
                html.append("<ul>"); in_list = True
            html.append(f"<li>{_strip_md(re.sub(r'^\s*([-*]|\d+\.)\s+', '', line))}</li>")
        else:
            if in_list:
                html.append("</ul>"); in_list = False
            html.append(f"<p>{_strip_md(line)}</p>")
    if in_list:
        html.append("</ul>")
    return "\n".join(html)


# --------------------------------------------------------------------------------------
# Factsheet (premium HTML)
# --------------------------------------------------------------------------------------
def _bignum(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1000:
        return f"{v / 1000:.0f}k"
    return f"{v:,.0f}"


def factsheet_html(f: dict, narrative_md: str) -> str:
    d5 = f.get("d5", {})
    d1 = f.get("d1", {})
    d2 = f.get("d2", {})
    logo = img_data_uri(C.LOGO_PATH)
    logo_html = (f"<img src='{logo}' style='height:58px'/>" if logo
                 else "<b style='font-size:22px;color:#0F5226'>NPHCDA</b>")

    # GAVI-style big-number stat blocks.
    stats = []
    if d5:
        stats.append((_bignum(d5.get("lga_total", 0)), C.ACCENT, "zero-dose children projected in 2026",
                      f"across {d5.get('lga_count', 0)} reporting LGAs"))
        stats.append((f"{d5.get('top20_pct', 0):.0f}%", C.NAVY, "of the burden sits in the top 20% of LGAs",
                      f"80% of the burden is in the top {d5.get('n80', 0)} LGAs"))
        stats.append((str(len(d5.get("tier1_states", []))), C.NPHCDA_GREEN, "Tier-1 critical states",
                      clean(", ".join(d5.get("tier1_states", [])) or "North-West")))
    stat_html = "".join(
        f"<div class='stat'><div class='big' style='color:{col}'>{clean(v)}</div>"
        f"<div class='lab'>{clean(lab)}</div><div class='cap'>{clean(cap)}</div></div>"
        for v, col, lab, cap in stats)

    # Pill callouts (GAVI-style rounded outline).
    pills = []
    if d5.get("top_lgas"):
        t = d5["top_lgas"][0]
        pills.append((f"{t['zd_count']:,}", f"highest-burden LGA: {t['lga']} ({t['state']}), {t['zd_rate_pct']:.0f}% rate"))
    if d5.get("top_states"):
        ts = d5["top_states"][0]
        pills.append((f"{ts['zd_2026_pct']:.0f}%", f"highest-risk state: {ts['state']} (predicted 2026)"))
    ar = d1.get("at_risk_antigens")
    pills.append((str(len(ar)) if ar else "0",
                  ("antigens at risk below 80% in 6-12m: " + ", ".join(ar)) if ar
                  else "antigens below the 80% target (all tracer antigens on track)"))
    if d2.get("top_drivers"):
        first = next(iter(d2["top_drivers"].items()))
        pills.append(("Drivers", f"{first[0]} dropout: {', '.join(first[1][:3])}"))
    pill_html = "".join(
        f"<div class='pill'><span class='pn'>{clean(n)}</span><span class='pt'>{clean(txt)}</span></div>"
        for n, txt in pills)

    rows = "".join(
        f"<tr><td>{i+1}</td><td>{clean(s['lga'])}</td><td>{clean(s['state'])}</td>"
        f"<td style='text-align:right'>{s['zd_count']:,}</td>"
        f"<td style='text-align:right'>{s['zd_rate_pct']:.0f}%</td></tr>"
        for i, s in enumerate(d5.get("top_lgas", [])))
    table_html = (f"<table class='tbl'><thead><tr><th>#</th><th>LGA</th><th>State</th>"
                  f"<th>Zero-dose children</th><th>Rate</th></tr></thead><tbody>{rows}</tbody></table>"
                  if rows else "")

    return f"""<!doctype html><html><head><meta charset='utf-8'>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Serif:wght@600;700&display=swap');
body{{font-family:'IBM Plex Sans',Segoe UI,sans-serif;color:#1A1A1A;margin:0;background:#fff}}
.wrap{{max-width:920px;margin:0 auto;padding:34px 42px}}
.hd{{display:flex;align-items:center;justify-content:space-between;border-bottom:4px solid {C.NPHCDA_GREEN};padding-bottom:14px}}
.tag{{color:{C.NPHCDA_GREEN};font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase}}
.lead{{font-family:'IBM Plex Serif',serif;font-size:25px;line-height:1.3;color:{C.NAVY};margin:18px 0 6px}}
.sub{{color:{C.MUTE};font-size:13px}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin:26px 0 8px;border-top:1px solid #e7eef4;padding-top:18px}}
.big{{font-family:'IBM Plex Serif',serif;font-size:46px;font-weight:700;line-height:1}}
.lab{{color:#1A1A1A;font-size:13px;margin-top:6px;max-width:15rem}}
.cap{{color:{C.MUTE};font-size:11.5px;margin-top:4px}}
.sect{{color:{C.NAVY};font-family:'IBM Plex Serif',serif;font-size:18px;border-left:4px solid {C.GOLD};padding-left:10px;margin:26px 0 10px}}
.pills{{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}}
.pill{{display:flex;align-items:center;gap:10px;border:1.5px solid {C.STEEL};border-radius:999px;padding:7px 14px;max-width:46%}}
.pn{{color:{C.STEEL};font-weight:700;font-size:15px;white-space:nowrap}}
.pt{{color:#33414d;font-size:11.5px;line-height:1.25}}
h2,h3{{color:{C.NAVY};font-family:'IBM Plex Serif',serif}} h2{{font-size:17px;margin-top:18px}} h3{{font-size:14px}}
ul{{margin:6px 0}} li{{margin:4px 0;font-size:13px}}
.tbl{{width:100%;border-collapse:collapse;margin-top:8px;font-size:12.5px}}
.tbl th{{background:{C.NAVY};color:#fff;padding:7px 9px;text-align:left}}
.tbl td{{padding:6px 9px;border-bottom:1px solid #e7eef4}}
.ft{{margin-top:26px;border-top:1px solid #e0e7ee;padding-top:10px;color:{C.MUTE};font-size:11px}}
</style></head><body><div class='wrap'>
<div class='hd'>{logo_html}<div style='text-align:right'><div class='tag'>Factsheet</div>
<div class='sub'>{clean(f.get('generated',''))}</div></div></div>
<div class='lead'>Nigeria zero-dose modelling: where the unvaccinated children are, and where to act first.</div>
<div class='sub'>{clean(f.get('consortium',''))}. For {clean(f.get('audience',''))}.</div>
<div class='stats'>{stat_html}</div>
<div class='sect'>Key statistics</div><div class='pills'>{pill_html}</div>
<div class='sect'>Findings and recommended actions</div>
{md_to_html(narrative_md)}
<div class='sect'>Highest-burden LGAs</div>{table_html}
<div class='ft'>Generated by the NPHCDA Zero-Dose Predictive Modelling Platform from live model outputs
(coverage, dropout and zero-dose workstreams). Figures are model estimates; review before use.
LGA population denominator: City Population (NPC 2022 projection).</div>
</div></body></html>"""


# --------------------------------------------------------------------------------------
# Policy brief (Word .docx)
# --------------------------------------------------------------------------------------
def policy_docx(f: dict, narrative_md: str) -> bytes:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    navy, green, gold, mute = RGBColor(0x1F, 0x3B, 0x57), RGBColor(0x1C, 0x7A, 0x3D), \
        RGBColor(0xC8, 0x90, 0x2A), RGBColor(0x6B, 0x7A, 0x88)
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    if C.LOGO_PATH.exists():
        try:
            doc.add_picture(str(C.LOGO_PATH), width=Inches(1.6))
        except Exception:
            pass
    t = doc.add_paragraph()
    r = t.add_run("Nigeria Zero-Dose Immunization - Policy Brief")
    r.bold = True; r.font.size = Pt(20); r.font.color.rgb = navy
    s = doc.add_paragraph()
    sr = s.add_run(f"{f.get('consortium','')}. For {f.get('audience','')}.  |  {f.get('generated','')}")
    sr.italic = True; sr.font.size = Pt(9); sr.font.color.rgb = mute

    for raw in narrative_md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("## "):
            h = doc.add_paragraph(); hr = h.add_run(_strip_md(line[3:]))
            hr.bold = True; hr.font.size = Pt(14); hr.font.color.rgb = navy
        elif line.startswith("### "):
            h = doc.add_paragraph(); hr = h.add_run(_strip_md(line[4:]))
            hr.bold = True; hr.font.size = Pt(12); hr.font.color.rgb = green
        elif re.match(r"^\s*[-*]\s+", line):
            doc.add_paragraph(_strip_md(re.sub(r"^\s*[-*]\s+", "", line)), style="List Bullet")
        elif re.match(r"^\s*\d+\.\s+", line):
            doc.add_paragraph(_strip_md(re.sub(r"^\s*\d+\.\s+", "", line)), style="List Number")
        else:
            doc.add_paragraph(_strip_md(line))

    d5 = f.get("d5", {})
    if d5.get("top_states"):
        h = doc.add_paragraph(); hr = h.add_run("Priority states (top 8 by risk)")
        hr.bold = True; hr.font.size = Pt(13); hr.font.color.rgb = navy
        tb = doc.add_table(rows=1, cols=4); tb.style = "Light Grid Accent 1"
        for i, c in enumerate(["State", "Zone", "ZD 2026 (%)", "ZD children 2026"]):
            cell = tb.rows[0].cells[i]; cr = cell.paragraphs[0].add_run(c); cr.bold = True
        for s in d5["top_states"]:
            cells = tb.add_row().cells
            cells[0].text = clean(s["state"]); cells[1].text = clean(s["zone"])
            cells[2].text = f"{s['zd_2026_pct']:.1f}"
            cells[3].text = f"{s['zd_2026_count']:,}" if s["zd_2026_count"] else "-"
    if d5.get("top_lgas"):
        h = doc.add_paragraph(); hr = h.add_run("Highest-burden LGAs (top 10)")
        hr.bold = True; hr.font.size = Pt(13); hr.font.color.rgb = navy
        tb = doc.add_table(rows=1, cols=4); tb.style = "Light Grid Accent 1"
        for i, c in enumerate(["LGA", "State", "ZD children", "Rate (%)"]):
            cr = tb.rows[0].cells[i].paragraphs[0].add_run(c); cr.bold = True
        for s in d5["top_lgas"]:
            cells = tb.add_row().cells
            cells[0].text = clean(s["lga"]); cells[1].text = clean(s["state"])
            cells[2].text = f"{s['zd_count']:,}"; cells[3].text = f"{s['zd_rate_pct']:.0f}"

    fp = doc.add_paragraph()
    fr = fp.add_run("Generated by the NPHCDA Zero-Dose Predictive Modelling Platform from live model "
                    "outputs (coverage, dropout and zero-dose workstreams). Figures are model estimates; review before use. "
                    "LGA population denominator: City Population (NPC 2022 projection).")
    fr.italic = True; fr.font.size = Pt(8); fr.font.color.rgb = mute

    buf = io.BytesIO(); doc.save(buf); return buf.getvalue()


def _section_lines(md: str, heads: list[str]) -> list[str]:
    """Return the bullet/numbered lines under any heading whose text matches one of `heads`."""
    out, cap = [], False
    for ln in md.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            cap = any(h.lower() in s.lower() for h in heads)
            continue
        if cap and s:
            out.append(_strip_md(re.sub(r"^\s*([-*]|\d+\.)\s*", "", s)))
    return out


def policy_pptx(f: dict, narrative_md: str) -> bytes:
    """Branded PowerPoint policy deck (NPHCDA logo, navy title bars)."""
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    NAVY, GREEN, GOLD, INK = RGBColor(0x1F, 0x3B, 0x57), RGBColor(0x1C, 0x7A, 0x3D), \
        RGBColor(0xC8, 0x90, 0x2A), RGBColor(0x1A, 0x1A, 0x1A)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    SW = Inches(13.333)

    def bar(slide, title):
        rect = slide.shapes.add_shape(1, 0, 0, SW, Inches(1.0))
        rect.fill.solid(); rect.fill.fore_color.rgb = NAVY; rect.line.fill.background()
        tf = rect.text_frame; tf.margin_left = Inches(0.4)
        tf.text = title
        p = tf.paragraphs[0]; p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255)
        accent = slide.shapes.add_shape(1, 0, Inches(1.0), SW, Inches(0.06))
        accent.fill.solid(); accent.fill.fore_color.rgb = GOLD; accent.line.fill.background()

    def bullets(slide, items, top=1.4, size=18):
        tb = slide.shapes.add_textbox(Inches(0.6), Inches(top), Inches(12.1), Inches(5.6))
        tf = tb.text_frame; tf.word_wrap = True
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = "- " + clean(it); p.font.size = Pt(size); p.font.color.rgb = INK; p.space_after = Pt(8)

    def table_slide(title, headers, rows):
        s = prs.slides.add_slide(blank); bar(s, title)
        n = len(rows) + 1
        tbl = s.shapes.add_table(n, len(headers), Inches(0.6), Inches(1.4),
                                 Inches(12.1), Inches(0.4 * n)).table
        for j, h in enumerate(headers):
            c = tbl.cell(0, j); c.text = h
            c.fill.solid(); c.fill.fore_color.rgb = NAVY
            c.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
            c.text_frame.paragraphs[0].font.bold = True; c.text_frame.paragraphs[0].font.size = Pt(13)
        for i, row in enumerate(rows, start=1):
            for j, val in enumerate(row):
                c = tbl.cell(i, j); c.text = str(val); c.text_frame.paragraphs[0].font.size = Pt(12)

    d5, d1, d2 = f.get("d5", {}), f.get("d1", {}), f.get("d2", {})

    # 1 - Title
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(1, 0, 0, SW, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
    if C.LOGO_PATH.exists():
        try:
            s.shapes.add_picture(str(C.LOGO_PATH), Inches(0.6), Inches(0.5), height=Inches(0.9))
        except Exception:
            pass
    tb = s.shapes.add_textbox(Inches(0.8), Inches(2.6), Inches(11.7), Inches(2.4)); tf = tb.text_frame
    tf.word_wrap = True
    tf.text = "Nigeria Zero-Dose Immunization"
    tf.paragraphs[0].font.size = Pt(40); tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    p = tf.add_paragraph(); p.text = "Policy Brief"
    p.font.size = Pt(28); p.font.color.rgb = GOLD
    p = tf.add_paragraph()
    p.text = f"{f.get('consortium', '')}.  For {f.get('audience', '')}.  |  {f.get('generated', '')}"
    p.font.size = Pt(13); p.font.color.rgb = RGBColor(220, 230, 238)

    # 2 - Headline numbers
    s = prs.slides.add_slide(blank); bar(s, "The headline")
    nums = []
    if d5:
        nums = [(f"{d5.get('lga_total', 0):,}", "zero-dose children, 2026"),
                (f"{d5.get('top20_pct', 0):.0f}%", "of burden in the top 20% of LGAs"),
                (str(len(d5.get('tier1_states', []))), "Tier-1 critical states")]
    for i, (big, lab) in enumerate(nums):
        x = Inches(0.6 + i * 4.2)
        tbx = s.shapes.add_textbox(x, Inches(2.2), Inches(4.0), Inches(2.6)); tf = tbx.text_frame
        tf.word_wrap = True
        tf.text = big
        tf.paragraphs[0].font.size = Pt(54); tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = [RGBColor(0xC0, 0x39, 0x2B), NAVY, GREEN][i]
        p = tf.add_paragraph(); p.text = clean(lab); p.font.size = Pt(16); p.font.color.rgb = INK

    # 3 - Key findings
    s = prs.slides.add_slide(blank); bar(s, "Key findings")
    kf = _section_lines(narrative_md, ["key findings"]) or [
        f"About {d5.get('lga_total', 0):,} zero-dose children across {d5.get('lga_count', 0)} LGAs in 2026.",
        "Tier-1 states: " + (", ".join(d5.get("tier1_states", [])) or "North-West") + "."]
    bullets(s, kf[:7])

    # 4 - Priority states
    if d5.get("top_states"):
        table_slide("Priority states (top 8 by risk)", ["State", "Zone", "ZD 2026 (%)", "ZD children 2026"],
                    [[clean(x["state"]), clean(x["zone"]), f"{x['zd_2026_pct']:.1f}",
                      f"{x['zd_2026_count']:,}" if x["zd_2026_count"] else "-"] for x in d5["top_states"]])
    # 5 - Priority LGAs
    if d5.get("top_lgas"):
        table_slide("Highest-burden LGAs (top 10)", ["LGA", "State", "ZD children", "Rate (%)"],
                    [[clean(x["lga"]), clean(x["state"]), f"{x['zd_count']:,}", f"{x['zd_rate_pct']:.0f}"]
                     for x in d5["top_lgas"]])

    # 6 - Recommendations
    s = prs.slides.add_slide(blank); bar(s, "Recommendations")
    recs = _section_lines(narrative_md, ["recommendation", "priority action", "implementation"]) or [
        f"Prioritize the top {d5.get('n80', 0)} LGAs that hold 80 percent of the burden.",
        "Concentrate first-line investment in the Tier-1 states.",
        "Address dropout with reminder-recall and defaulter tracing.",
        "Use the LGA hotspot map to micro-plan supervision."]
    bullets(s, recs[:7])
    foot = s.shapes.add_textbox(Inches(0.6), Inches(6.9), Inches(12.1), Inches(0.5))
    fp = foot.text_frame; fp.text = ("Generated by the NPHCDA Zero-Dose Predictive Modelling Platform "
                                     "from live model outputs. Figures are model estimates.")
    fp.paragraphs[0].font.size = Pt(9); fp.paragraphs[0].font.color.rgb = RGBColor(0x6B, 0x7A, 0x88)

    buf = io.BytesIO(); prs.save(buf); return buf.getvalue()
