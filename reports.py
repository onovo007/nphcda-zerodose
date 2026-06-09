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
def factsheet_html(f: dict, narrative_md: str) -> str:
    d5 = f.get("d5", {})
    logo = img_data_uri(C.LOGO_PATH)
    logo_html = f"<img src='{logo}' style='height:54px'/>" if logo else "<b style='font-size:20px;color:#0F5226'>NPHCDA</b>"
    kpis = []
    if d5:
        kpis = [
            ("Zero-dose children, 2026", f"{d5.get('lga_total', 0):,}"),
            ("Reporting LGAs", f"{d5.get('lga_count', 0)}"),
            ("Burden in top 20% of LGAs", f"{d5.get('top20_pct', 0):.0f}%"),
            ("Tier-1 critical states", str(len(d5.get('tier1_states', [])))),
        ]
    kpi_html = "".join(
        f"<div class='k'><div class='kl'>{clean(l)}</div><div class='kv'>{clean(v)}</div></div>"
        for l, v in kpis)
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
.wrap{{max-width:900px;margin:0 auto;padding:34px 40px}}
.hd{{display:flex;align-items:center;justify-content:space-between;border-bottom:4px solid {C.NPHCDA_GREEN};padding-bottom:14px}}
.title{{font-family:'IBM Plex Serif',serif;font-size:26px;color:{C.NAVY};margin:14px 0 2px}}
.sub{{color:{C.MUTE};font-size:13px}}
.kgrid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}}
.k{{background:linear-gradient(180deg,#fff,#f5f9f6);border:1px solid rgba(28,122,61,.18);border-top:4px solid {C.NPHCDA_GREEN};border-radius:12px;padding:12px 14px}}
.kl{{color:{C.MUTE};font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px}}
.kv{{color:{C.NAVY};font-size:24px;font-weight:700}}
h2{{color:{C.NAVY};font-family:'IBM Plex Serif',serif;border-left:4px solid {C.GOLD};padding-left:10px;font-size:18px;margin-top:22px}}
h3{{color:{C.NAVY};font-size:15px}}
ul{{margin:6px 0 6px 0}} li{{margin:4px 0}}
.tbl{{width:100%;border-collapse:collapse;margin-top:8px;font-size:13px}}
.tbl th{{background:{C.NAVY};color:#fff;padding:7px 9px;text-align:left}}
.tbl td{{padding:6px 9px;border-bottom:1px solid #e7eef4}}
.ft{{margin-top:24px;border-top:1px solid #e0e7ee;padding-top:10px;color:{C.MUTE};font-size:11px}}
</style></head><body><div class='wrap'>
<div class='hd'>{logo_html}<div style='text-align:right'><div class='sub'>FACTSHEET - {clean(f.get('generated',''))}</div></div></div>
<div class='title'>Nigeria Zero-Dose Immunization - Key Findings</div>
<div class='sub'>{clean(f.get('consortium',''))}. For {clean(f.get('audience',''))}.</div>
<div class='kgrid'>{kpi_html}</div>
{md_to_html(narrative_md)}
<h2>Highest-burden LGAs</h2>{table_html}
<div class='ft'>Generated by the NPHCDA Zero-Dose Predictive Modelling Platform from live model outputs
(Domains 1, 2 and 5). Figures are model estimates; review before use. LGA population denominator:
City Population (NPC 2022 projection).</div>
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
                    "outputs (Domains 1, 2 and 5). Figures are model estimates; review before use. "
                    "LGA population denominator: City Population (NPC 2022 projection).")
    fr.italic = True; fr.font.size = Pt(8); fr.font.color.rgb = mute

    buf = io.BytesIO(); doc.save(buf); return buf.getvalue()
