"""Domain 1 publication figures from the corrected forecast outputs (PNG 300 dpi, SVG, PDF + source data)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "corrected"
FIG = ROOT / "figures"; (FIG / "data").mkdir(parents=True, exist_ok=True)
DATA = ROOT.parent / "All I Need" / "03_Datasets" / "datasets"
GEO = ROOT.parent / "Domain5_Update_20261009" / "Updated Datasets for Domain 5" / "LGA_Archetypes"
NAVY, GREY, INK = "#1F3B57", "#5B6B79", "#27343D"
AC = {"BCG": "#1565C0", "Penta1": "#2E7D32", "Penta3": "#E65100", "Measles1": "#880E4F"}
SEV = {"Collapsed (<0%)": "#4A0D0D", "Critical (0-40%)": "#A50026", "Severe (40-70%)": "#D73027", "Warning (70-80%)": "#F46D43"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9AA5B1", "axes.titleweight": "bold", "axes.titlecolor": NAVY, "savefig.dpi": 300,
                     "svg.fonttype": "none"})
CAP = {}


def save(fig, name, cap, data=None, note=None):
    if note:
        fig.text(0.01, -0.02, note, fontsize=7.5, color=GREY, style="italic", ha="left", va="top")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    if data is not None:
        data.to_csv(FIG / "data" / f"{name}.csv", index=False)
    CAP[name] = cap


def clean(s):
    s = str(s).strip()
    for suf in (" Local Government Area", " LGA"):
        if s.lower().endswith(suf.lower()):
            s = s[: -len(suf)]
    p = s.split(None, 1)
    return (p[1] if len(p) > 1 and len(p[0]) == 2 and p[0].islower() else s).strip()


nat_sum = pd.read_csv(OUT / "d1_national_at_risk_summary.csv").set_index("antigen")
nat = pd.read_csv(OUT / "d1_national_monthly_doses.csv", parse_dates=["ds"])
cols = {"BCG": "bcg_count", "Penta1": "penta_1_count", "Penta3": "penta_3_count", "Measles1": "measles_1_count"}

# F1 national forecasts (% of 2024 baseline)
fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
for ax, (a, c) in zip(axs.flat, cols.items()):
    f = pd.read_csv(OUT / f"d1_national_forecast_{a}.csv", parse_dates=["ds"])
    base = f.base_2024.iloc[0]; cut = nat.ds.max()
    obs = nat[["ds", c]]
    h, fo = f[f.ds <= cut], f[f.ds > cut]
    ax.scatter(obs.ds, obs[c] / base * 100, s=12, color=AC[a], alpha=0.7, label="Observed")
    ax.plot(h.ds, h.yhat / base * 100, color=AC[a], lw=1.8, label="Fitted")
    ax.plot(fo.ds, fo.yhat / base * 100, "--", color=AC[a], lw=2, label="Forecast")
    ax.fill_between(fo.ds, fo.yhat_lower / base * 100, fo.yhat_upper / base * 100, color=AC[a], alpha=0.15, label="95% interval")
    ax.axhline(80, color="#C0392B", ls="--", lw=1.4, label="80% of 2024 level")
    ax.axvline(cut, color="#888", ls=":", lw=1)
    r = nat_sum.loc[a]
    ax.set_title(f"{a}: lowest forecast {r.min_forecast_pct:.0f}% of 2024 ({r.min_when})", loc="left", fontsize=10.5)
    ax.set_ylim(0, max(160, ax.get_ylim()[1])); ax.set_ylabel("% of 2024 monthly doses")
axs[0, 0].legend(fontsize=7.5, ncol=2, frameon=False, loc="lower left")
save(fig, "D1_F1_national_tracer_forecasts", "National monthly doses of the four tracer antigens as % of their 2024 level, with Prophet forecasts to mid-2027.",
     nat, "DHIS2 2021-2025, all 774 LGAs. Katsina, Kogi, Kwara, Lagos and Niger have no rows before January 2023. Model estimate.")

# F2 cohort coverage 2026 vs NDHS 2024
u = pd.read_csv(DATA / "under_5_2024.csv").iloc[1:]; u.columns = ["z", "s", "u5"]
coh = (u.u5.astype(str).str.replace(",", "").astype(float) / 5).round(0).sum()
nd = pd.read_csv(DATA.parent.parent.parent / "Domain5_Update_20261009" / "Updated Datasets for Domain 5" / "Method1_Bayesian_Hierarchical_Model" / "under_5_2024.csv").iloc[1:]
svy_src = pd.read_csv(Path(r"C:\Users\dramo\OneDrive\Desktop\sinbad\nphcda-zerodose-git\data\sample\ndhs_antigens2024.csv"))
u2 = u.assign(state=u.s.str.strip().str.replace(", Abuja", "").str.replace("FCT, ABUJA", "FCT"), w=u.u5.astype(str).str.replace(",", "").astype(float))
svy_src["key"] = svy_src.State.str.strip().str.upper().str.replace(" ", "")
u2["key"] = u2.s.str.strip().str.upper().str.replace(" ", "").str.replace(",ABUJA", "")
sv = svy_src.merge(u2[["key", "w"]], on="key", how="left")
scol = {"BCG": "BCG vaccination received", "Penta1": "Pentavalent 1 vaccination received", "Penta3": "Pentavalent 3 vaccination received",
        "Measles1": "Measles vaccination received"}
rows = []
for a in cols:
    f = pd.read_csv(OUT / f"d1_national_forecast_{a}.csv", parse_dates=["ds"])
    adm = (f[f.ds.dt.year == 2026].yhat / (coh / 12) * 100).mean()
    m = sv[scol[a]].notna() & sv.w.notna()
    svy = (sv.loc[m, scol[a]] * sv.loc[m, "w"]).sum() / sv.loc[m, "w"].sum()
    rows.append(dict(antigen=a, admin_coverage_2026_pct=adm, ndhs_2024_pct=svy, matched_states=int(m.sum())))
cv = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(4); w = 0.38
b1 = ax.bar(x - w / 2, cv.admin_coverage_2026_pct, w, color="#2E6E8E", label="Administrative coverage 2026 (forecast doses / eligible cohort)")
b2 = ax.bar(x + w / 2, cv.ndhs_2024_pct, w, color="#C68E2B", label="NDHS 2024 survey coverage")
for bars in (b1, b2):
    for bb in bars:
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 2, f"{bb.get_height():.0f}%", ha="center", fontsize=10, fontweight="bold", color=NAVY)
ax.axhline(100, color="#7F8C97", ls=":", lw=1.2); ax.axhline(80, color="#C0392B", ls="--", lw=1.4)
ax.text(3.45, 81.5, "80% target", color="#C0392B", fontsize=9, ha="right"); ax.text(3.45, 101.5, "100%", color="#7F8C97", fontsize=9, ha="right")
ax.set_xticks(x); ax.set_xticklabels(list(cols)); ax.set_ylabel("Coverage, %"); ax.set_ylim(0, max(cv.admin_coverage_2026_pct) * 1.32)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)
ax.set_title("Routine data imply coverage above 100%; the surveys do not", loc="left")
save(fig, "D1_F2_cohort_coverage_vs_ndhs", "2026 administrative coverage of the eligible cohort (forecast monthly doses / (12-23-month cohort / 12)) against NDHS 2024 survey coverage.",
     cv, f"Eligible cohort = 2024 under-five projection / 5 ({coh / 1e6:.2f} million). NDHS 2024 national values weighted by state under-five population. Model estimate.")

# F3 LGA early-warning composite
fa = pd.read_csv(OUT / "d1_lga_forecast_all.csv")
ar = fa[fa.crosses_80_in_6_12m].copy()
ar["severity"] = pd.cut(ar.min_forecast_pct_6_12m, [-np.inf, 0, 40, 70, 80], labels=list(SEV))
ar["lga_clean"] = ar.lga.map(clean)
fig = plt.figure(figsize=(17, 12.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.6, 2.2], hspace=0.55, wspace=0.75)
for j, a in enumerate(cols):
    sub = ar[ar.antigen == a]
    ax = fig.add_subplot(gs[0, j]); sc = sub.severity.value_counts().reindex(list(SEV), fill_value=0)
    ax.bar(range(4), sc.values, color=list(SEV.values()))
    for i, v in enumerate(sc.values):
        ax.text(i, v + 2, str(v), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(["<0%", "0-40", "40-70", "70-80"], fontsize=8.5)
    ax.set_title(f"{a}: {len(sub)} of {int((fa.antigen == a).sum())} LGAs flagged", color=AC[a], fontsize=11)
    ax.set_ylabel("LGAs")
    ax2 = fig.add_subplot(gs[1, j]); st = sub[sub.min_forecast_pct_6_12m >= 0].groupby("state").size().sort_values().tail(8)
    ax2.barh(range(len(st)), st.values, color="#475569")
    ax2.set_yticks(range(len(st))); ax2.set_yticklabels(st.index, fontsize=9); ax2.set_title("Top 8 states by flagged LGAs", loc="left", fontsize=9.5)
    for i, v in enumerate(st.values):
        ax2.text(v + 0.3, i, str(v), va="center", fontsize=8.5)
    ax3 = fig.add_subplot(gs[2, j]); cr = sub[sub.min_forecast_pct_6_12m >= 0].sort_values("min_forecast_pct_6_12m")
    top = pd.concat([cr[cr.min_forecast_pct_6_12m < 40].head(3), cr[cr.min_forecast_pct_6_12m.between(40, 70, inclusive="left")].head(4),
                     cr[cr.min_forecast_pct_6_12m >= 70].head(3)]).iloc[::-1]
    for i, r in enumerate(top.itertuples()):
        ax3.barh(i, 80 - r.min_forecast_pct_6_12m, left=r.min_forecast_pct_6_12m, color=SEV[str(r.severity)], height=0.55)
        ax3.text(81.5, i, f"{r.min_forecast_pct_6_12m:.0f}%", va="center", ha="left", fontsize=8, color=SEV[str(r.severity)], fontweight="bold")
    ax3.axvline(80, color="#A50026", ls="--", lw=1.3)
    ax3.set_yticks(range(len(top))); ax3.set_yticklabels([f"{r.lga_clean.replace(' Area Council', '')[:20]} ({r.state})" for r in top.itertuples()], fontsize=8.5)
    ax3.set_xlim(0, 95); ax3.set_xlabel("Lowest forecast in months 6-12, % of 2024", fontsize=8.5)
    ax3.set_title("Illustrative flagged LGAs", loc="left", fontsize=9.5)
fig.legend(handles=[Patch(color=c, label=k) for k, c in SEV.items()], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("LGA early-warning: tracer antigens projected below 80% of their 2024 level within 6-12 months", x=0.01, ha="left", fontsize=13,
             fontweight="bold", color=NAVY)
save(fig, "D1_F3_lga_early_warning_composite", "LGA early-warning flags by antigen: severity, states with most flagged LGAs, and illustrative LGAs.",
     ar.drop(columns=["lga"]), "Prophet forecasts per LGA and antigen, DHIS2 2021-2025, 773 LGAs. Severity = lowest forecast in months 6-12 after Dec 2025. Model estimate.")

# F4 map: number of tracer antigens flagged per LGA
import geopandas as gpd  # noqa: E402
keys = pd.read_csv(GEO / "lga_map_keys_774.csv")
g = gpd.read_file(GEO / "nga_lgas.geojson").rename(columns={"lga_key": "geo_lga_key"}).drop(columns=["state", "lga"])
g = g.merge(keys[["lga_uid", "state", "lga", "state_key", "geo_lga_key"]], on=["state_key", "geo_lga_key"], how="left")
gs_ = gpd.read_file(GEO / "nga_states.geojson")
import re  # noqa: E402
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", str(s).lower())).strip()
cnt = fa.assign(k=fa.lga.map(clean).map(norm), s=fa.state).groupby(["s", "k"]).crosses_80_in_6_12m.sum().rename("n").reset_index()
g["k"] = g.lga.map(norm)
g = g.merge(cnt, left_on=["state", "k"], right_on=["s", "k"], how="left")
miss = g.n.isna().sum()
fig, ax = plt.subplots(figsize=(9, 8))
cmap = {0: "#E8EEF3", 1: "#FDD49E", 2: "#FC8D59", 3: "#E34A33", 4: "#7F0000"}
for k_, c in cmap.items():
    sub = g[g.n == k_]
    if len(sub):
        sub.plot(ax=ax, color=c, edgecolor="white", linewidth=0.1)
g[g.n.isna()].plot(ax=ax, color="#BBBBBB", hatch="///", edgecolor="#777", linewidth=0.1)
gs_.boundary.plot(ax=ax, color="#3A4A5A", linewidth=0.45); ax.set_axis_off()
vc = g.n.value_counts()
ax.legend(handles=[Patch(color=c, label=f"{k_} of 4 antigens ({int(vc.get(k_, 0))} LGAs)") for k_, c in cmap.items()] +
          [Patch(color="#BBBBBB", label=f"No forecast ({int(miss)})")], loc="lower right", fontsize=8, frameon=False, title="Antigens flagged", title_fontsize=8.5)
ax.set_title("Number of tracer antigens with an early-warning flag, by LGA", loc="left", fontsize=11)
save(fig, "D1_F4_lga_flag_map", "Number of the four tracer antigens (BCG, Penta1, Penta3, Measles1) projected below 80% of the LGA's 2024 level within 6-12 months.",
     g[["state", "lga", "n"]], "GRID3 boundaries. Model estimate.")
CAP["_map_unmatched"] = int(miss)

# F5 additional antigens
sa = pd.read_csv(OUT / "D1_additional_antigen_series.csv", parse_dates=["ds"])
ss = pd.read_csv(OUT / "D1_additional_antigen_forecast_summary.csv").set_index("Antigen")
fig, axs = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
for ax, a in zip(axs.flat, ss.index):
    s = sa[sa.antigen == a]; o, f = s[s.kind == "observed"], s[s.kind == "forecast"]
    est = ss.loc[a, "Group"] == "established"
    ax.plot(o.ds, o.y / 1000, color=NAVY, lw=1.8, label="Observed")
    ax.plot(f.ds, f.y / 1000, "--", color="#C0392B", lw=1.8, label="Forecast")
    ax.fill_between(f.ds, f.yhat_lower / 1000, f.yhat_upper / 1000, color="#C0392B", alpha=0.15)
    if est:
        ax.axhline(s.base_2024.iloc[0] * 0.8 / 1000, color="#7B2FBF", ls=":", lw=1.6, label="80% of 2024 level")
    ax.set_title(f"{a} ({'established' if est else 'recently introduced'}): min {ss.loc[a, 'Min forecast % of 2024']:.0f}% of 2024", loc="left", fontsize=10)
    ax.set_ylabel("Doses per month (000s)", fontsize=8.5)
axs[0, 0].legend(fontsize=7.5, frameon=False)
save(fig, "D1_F5_additional_antigens", "Forecasts for nine additional antigens (DHIS2 2021-2025, projected 30 months).", ss.reset_index(),
     "Established antigens screened against 80% of their own 2024 level; recently introduced antigens read as uptake. Model estimate.")

# F6 OPV3 worked example
s = sa[sa.antigen == "OPV3"]; o, f = s[s.kind == "observed"], s[s.kind == "forecast"]; base = s.base_2024.iloc[0]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(o.ds, o.y / 1000, color=NAVY, lw=2, label="Observed OPV3 doses")
ax.plot(f.ds, f.y / 1000, "--", color="#C0392B", lw=2, label="Forecast"); ax.fill_between(f.ds, f.yhat_lower / 1000, f.yhat_upper / 1000, color="#C0392B", alpha=0.15)
ax.axhline(base * 0.8 / 1000, color="#7B2FBF", ls=":", lw=2, label=f"80% of 2024 level ({base * 0.8 / 1000:.0f}k per month)")
ax.set_ylabel("Doses per month (000s)"); ax.legend(frameon=False, fontsize=8.5, loc="lower left")
ax.set_title(f"OPV3: lowest forecast {ss.loc['OPV3', 'Min forecast % of 2024']:.0f}% of the 2024 level, so no early-warning", loc="left", fontsize=11)
save(fig, "D1_F6_opv3_worked_example", "Worked example: the dotted line is 80% of OPV3's own 2024 monthly doses; the forecast stays above it.", None)
(FIG / "captions.json").write_text(json.dumps(CAP, indent=2))
print(cv.round(1).to_string()); print("unmatched map LGAs:", miss); print(sorted(p.name for p in FIG.glob("*.png")))
