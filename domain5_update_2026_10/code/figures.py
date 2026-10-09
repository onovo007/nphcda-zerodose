"""Publication figures for the Domain 5 update (deck, report, abstracts). PNG (300 dpi), SVG, PDF + plot data.

Method 1 = Bayesian hierarchical Beta regression with DHIS2-calibrated LGA allocation (773 LGAs).
Method 2 = Bayesian small-area estimation (SAE) (774 LGAs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent / "worldbank_lsms_2026_domain5_20261008"
sys.path.insert(0, str(WS / "05_code"))
import d5config as C  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
(FIG / "data").mkdir(parents=True, exist_ok=True)
AF = C.ARCH_FILES / "AIN"
SAE = WS / "16_SAE_joint_bayesian_model"

NAVY, STEEL, GREY, INK, LIGHT = "#1F3B57", "#2E6E8E", "#5B6B79", "#27343D", "#EEF2F6"
M1C, M2C = "#0072B2", "#D55E00"
ARC = {1: "#7F1D1D", 2: "#D55E00", 3: "#56B4E9", 4: "#E6A817", 5: "#1C7A3D"}
ARCN = {1: "Remote Rural / Hard-to-Reach", 2: "Conflict-Affected / Nomadic", 3: "Riverine / Geographically Isolated",
        4: "Peri-urban / Migrant Dense", 5: "Urban Slums (better-off core)"}
HOT = {"Hot Spot (p<0.01)": "#B2182B", "Hot Spot (p<0.05)": "#EF8A62", "Hot Spot (p<0.10)": "#FDDBC7",
       "Not Significant": "#E3E3E3", "Cold Spot (p<0.10)": "#D1E5F0", "Cold Spot (p<0.05)": "#67A9CF",
       "Cold Spot (p<0.01)": "#2166AC"}
M1N = "Method 1: Bayesian hierarchical model with DHIS2-calibrated LGA allocation"
M2N = "Method 2: Bayesian small-area estimation"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9AA5B1", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 300, "svg.fonttype": "none",
                     "axes.titleweight": "bold", "axes.titlecolor": NAVY})
CAP = {}
from matplotlib.ticker import FuncFormatter  # noqa: E402
KFMT = FuncFormatter(lambda x, _: f"{x:,.0f}")


def save(fig, name, caption, data=None, source=None):
    if source:
        fig.text(0.01, -0.02, source, fontsize=8, color=GREY, style="italic", ha="left", va="top", wrap=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    if data is not None:
        data.to_csv(FIG / "data" / f"{name}.csv", index=False)
    CAP[name] = caption


def hotspot_class(z, p):
    if p <= 0.01: return ("Hot" if z > 0 else "Cold") + " Spot (p<0.01)"
    if p <= 0.05: return ("Hot" if z > 0 else "Cold") + " Spot (p<0.05)"
    if p <= 0.10: return ("Hot" if z > 0 else "Cold") + " Spot (p<0.10)"
    return "Not Significant"


def gi(g, col, kind="knn"):
    from esda.getisord import G_Local
    from libpysal.weights import KNN, Queen
    w = KNN.from_dataframe(g, k=5) if kind == "knn" else Queen.from_dataframe(g, use_index=False)
    w.transform = "r"
    r = G_Local(g[col].values.astype(float), w, star=True, seed=42)
    return [hotspot_class(z, p) for z, p in zip(r.Zs, r.p_sim)]


def load_geo():
    import geopandas as gpd
    g = gpd.read_file(AF / "claude_model/data/sample/geo/nga_lgas.geojson").rename(columns={"lga_key": "geo_lga_key"}).drop(columns=["state", "lga"])
    xw = pd.read_parquet(C.D["integ"] / "domain5_all774_lga_geographic_crosswalk.parquet")
    xw["geo_lga_key"] = np.where(xw.grid3_match == "exact", xw.lga_key, xw.grid3_match.str.split(":", n=1).str[-1])
    g = g.merge(xw[["state_key", "geo_lga_key", "state", "lga_clean"]], on=["state_key", "geo_lga_key"], how="left")
    gs = gpd.read_file(AF / "claude_model/data/sample/geo/nga_states.geojson")
    return g, gs


def basemap(ax, g, gs):
    ax.set_axis_off()
    gs.boundary.plot(ax=ax, color="#3A4A5A", linewidth=0.45, zorder=5)


def main():
    lg = pd.read_csv(RES / "lga_both_methods_774.csv")
    st = pd.read_csv(RES / "state_both_methods.csv")
    zn = pd.read_csv(RES / "national_zone_both_methods.csv")
    val = pd.read_csv(RES / "validation_both_methods.csv")
    curves = pd.read_csv(RES / "pareto_curves.csv")
    pareto = pd.read_csv(RES / "pareto_scenarios.csv")
    arch = pd.read_csv(RES / "archetype_both_methods.csv")
    R = json.loads((RES / "results.json").read_text())
    g, gs = load_geo()
    g = g.merge(lg, on=["state", "lga_clean"], how="left")
    assert g.m2_rate.notna().sum() == 774

    # ===================== Method 1 figures =====================
    # M1-1: LGA rate map + LGA hotspots (Gi*, k=5)
    gm = g.copy()
    gm["m1_fill"] = gm.m1_rate.fillna(gm.m1_rate.median())
    gm["gi"] = gi(gm, "m1_fill")
    gm.loc[gm.m1_rate.isna(), "gi"] = "Not estimated"
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    gm.plot(ax=axs[0], column="m1_rate", cmap="YlOrRd", vmin=0, vmax=99, edgecolor="white", linewidth=0.08,
            legend=True, legend_kwds={"shrink": 0.6, "label": "Zero-dose rate, % of children 12-23 months"},
            missing_kwds={"color": "#BBBBBB", "hatch": "///", "edgecolor": "#666", "label": "Not estimated"})
    basemap(axs[0], gm, gs); axs[0].set_title("A. Zero-dose rate by LGA, 2026", loc="left", fontsize=11)
    for k, c in {**HOT, "Not estimated": "#BBBBBB"}.items():
        sub = gm[gm.gi == k]
        if len(sub):
            sub.plot(ax=axs[1], color=c, edgecolor="white", linewidth=0.08)
    basemap(axs[1], gm, gs)
    cnt = gm.gi.value_counts()
    axs[1].legend(handles=[Patch(color=c, label=f"{k} ({cnt.get(k, 0)})") for k, c in HOT.items() if cnt.get(k, 0)],
                  fontsize=7.5, frameon=False, loc="lower right")
    axs[1].set_title("B. Hotspot clusters (Getis-Ord Gi*, k = 5)", loc="left", fontsize=11)
    save(fig, "M1_01_lga_rate_and_hotspots",
         "Method 1 LGA zero-dose rate for 2026 and Getis-Ord Gi* hotspot clusters, 773 modelled LGAs; Guzamala (Borno) not estimated.",
         gm[["state", "lga_clean", "m1_rate", "gi"]],
         "Model estimate. Method 1, 2026. Sources: NDHS 2008-2024; DHIS2 Penta1 2021-2025; NPC 2022; GRID3 boundaries. n = 773 LGAs.")

    # M1-2: state hotspot persistence 2026-2028 (Queen contiguity)
    s1 = pd.read_parquet(C.D["bayes"] / "D_corrected_balanced_trend" / "state_results_original_format.parquet")
    gsv = gs.copy()
    from d5lib import N
    s1["state_key"] = s1.state.map(N.nstate)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9))
    dat = []
    for ax, yr in zip(axs, (2026, 2027, 2028)):
        t = gsv.merge(s1[["state_key", f"zd_pred_{yr}_mean"]], on="state_key", how="left")
        t["gi"] = gi(t, f"zd_pred_{yr}_mean", "queen")
        for k, c in HOT.items():
            sub = t[t.gi == k]
            if len(sub):
                sub.plot(ax=ax, color=c, edgecolor="white", linewidth=0.6)
        ax.set_axis_off(); ax.set_title(str(yr), fontsize=12)
        dat.append(t.drop(columns="geometry").assign(year=yr))
    fig.legend(handles=[Patch(color=c, label=k) for k, c in HOT.items()], loc="lower center", ncol=7, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    save(fig, "M1_02_state_hotspots_2026_2028",
         "Forecast state zero-dose hotspots 2026-2028 (Getis-Ord Gi*, Queen contiguity) from the Method 1 Bayesian state model.",
         pd.concat(dat)[["state", "year", "gi"]],
         "Model estimate. Method 1 state posterior means 2026-2028; GRID3 admin-1 boundaries.")

    # M1-3: Pareto with 50/60/80 markers (Method 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    b = np.sort(lg.m1_children.dropna().values)[::-1]
    ax.bar(np.arange(1, len(b) + 1), b, width=1.0, color=np.where(np.arange(len(b)) < R["m1"]["pareto"]["50"], "#C0392B",
                                                               np.where(np.arange(len(b)) < R["m1"]["pareto"]["80"], STEEL, "#B8C4CF")))
    ax.set_ylabel("Modelled zero-dose children, 2026"); ax.set_xlabel("LGAs ranked by modelled burden (773)"); ax.yaxis.set_major_formatter(KFMT)
    ax2 = ax.twinx(); c = np.cumsum(b) / b.sum() * 100
    ax2.plot(np.arange(1, len(b) + 1), c, color=NAVY, lw=2.2); ax2.set_ylim(0, 102); ax2.set_ylabel("Cumulative share of burden (%)")
    ax2.spines["right"].set_visible(True)
    for s_, k in R["m1"]["pareto"].items():
        ax2.axhline(int(s_), color="#9AA5B1", lw=0.7, ls=":"); ax2.plot([k], [int(s_)], "o", color=NAVY)
        ax2.annotate(f"{s_}% in {k} LGAs", (k, int(s_)), xytext=(10, -14), textcoords="offset points", fontsize=9.5,
                     color=NAVY, fontweight="bold")
    ax.set_title("Pareto concentration of zero-dose burden, Method 1", loc="left")
    save(fig, "M1_03_pareto", "Pareto concentration of modelled 2026 zero-dose burden across 773 LGAs (Method 1), with the number of LGAs needed to reach 50%, 60% and 80%.",
         pd.DataFrame({"rank": np.arange(1, len(b) + 1), "children": b, "cum_pct": c}),
         f"Model estimate. Method 1; national total {R['m1']['national']:,.0f} children aged 12-23 months.")

    # M1-4: NmDHS validation (zone bars + state scatter)
    zv = pd.read_csv(RES / "zone_validation_nmdhs.csv")
    v1 = val[(val.method == "Method 1") & val.check.str.startswith("NmDHS")].iloc[0]
    fig, (a, b2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    zz = zv.sort_values("nmdhs_2025_26", ascending=False); x = np.arange(len(zz)); w = 0.38
    a.bar(x - w / 2, zz.m1_state_model_rate, w, color=M1C, label="Method 1 forecast, 2026")
    a.bar(x + w / 2, zz.nmdhs_2025_26, w, color="#9AA5B1", label="NmDHS 2025-26 survey")
    for xi, (m, n) in enumerate(zip(zz.m1_state_model_rate, zz.nmdhs_2025_26)):
        a.text(xi - w / 2, m + 0.8, f"{m:.0f}", ha="center", fontsize=9, color=M1C)
        a.text(xi + w / 2, n + 0.8, f"{n:.0f}", ha="center", fontsize=9, color=GREY)
    a.set_xticks(x); a.set_xticklabels([z.replace(" ", "\n") for z in zz.area]); a.set_ylabel("Zero-dose, % (100 - Penta1)")
    a.legend(frameon=False); a.set_title("A. Zone level", loc="left")
    b2.scatter(st.nmdhs_2025_26, st.m1_rate, s=36, color=M1C, edgecolor="white", zorder=3)
    b2.plot([0, 85], [0, 85], ls="--", color="#9AA5B1")
    for _, r in st.iterrows():
        if r.state in ("Sokoto", "Kebbi", "Zamfara", "Kano", "Niger", "Lagos", "Abia"):
            b2.annotate(r.state, (r.nmdhs_2025_26, r.m1_rate), fontsize=8, xytext=(3, 3), textcoords="offset points")
    b2.set_xlabel("NmDHS 2025-26 zero-dose, %"); b2.set_ylabel("Method 1 forecast 2026, %")
    b2.set_title(f"B. 37 states: Spearman rho = {v1.rho:.2f} (95% CI {v1.ci_lo:.2f}-{v1.ci_hi:.2f}), p < 0.001", loc="left", fontsize=10)
    save(fig, "M1_04_nmdhs_validation", "Method 1 state forecasts for 2026 against the Nigeria mini Demographic and Health Survey 2025-26 (not used in fitting).",
         st[["state", "nmdhs_2025_26", "m1_rate"]],
         f"Model estimate vs survey. MAE {v1.MAE_pp:.1f} percentage points. Sources: NmDHS 2025-26 Table 10 (100 - Penta1); Method 1 forecast.")

    # M1-5: IHME concordance (terciles + scatter)
    d = lg.dropna(subset=["m1_rate", "ihme_zd_2018"]).copy()
    d["t_m"] = pd.qcut(d.m1_rate.rank(method="first"), 3, labels=[0, 1, 2]).astype(int)
    d["t_i"] = pd.qcut(d.ihme_zd_2018.rank(method="first"), 3, labels=[0, 1, 2]).astype(int)
    ct = pd.crosstab(d.t_m, d.t_i)
    opp = int(ct.loc[0, 2] + ct.loc[2, 0]); same = int(np.trace(ct.values))
    io = val[(val.method == "Method 1") & (val.check == "IHME DTP1 2018, LGA overall")].iloc[0]
    fig, (a, b2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    labs = ["Lower", "Middle", "Higher"]
    for i in range(3):
        for j in range(3):
            dd = abs(i - j); col = "#CFE8D5" if dd == 0 else "#F6E7C8" if dd == 1 else "#F4CCCC"
            a.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color=col, ec="white"))
            a.text(j, i, int(ct.iloc[i, j]), ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(labs); a.set_yticklabels(labs)
    a.set_xlim(-.5, 2.5); a.set_ylim(-.5, 2.5); a.set_xlabel("IHME 2018 zero-dose tercile"); a.set_ylabel("Method 1 2026 zero-dose tercile")
    a.set_title(f"A. Tercile agreement: {same} same, {opp} opposite", loc="left")
    b2.scatter(d.ihme_zd_2018, d.m1_rate, s=9, alpha=0.55, color=M1C, edgecolor="none")
    b2.set_xlabel("IHME 2018 zero-dose (100 - DTP1), %"); b2.set_ylabel("Method 1 2026 zero-dose, %")
    b2.set_title(f"B. {io.n} LGAs: Spearman rho = {io.rho:.2f} (95% CI {io.ci_lo:.2f}-{io.ci_hi:.2f})", loc="left", fontsize=10)
    save(fig, "M1_05_ihme_concordance", "Method 1 LGA estimates (2026) against the IHME Local Burden of Disease DTP1 surface (2018).",
         d[["state", "lga_clean", "m1_rate", "ihme_zd_2018", "t_m", "t_i"]],
         "Cross-year comparison (2026 model vs 2018 IHME). Sources: IHME LBD DTP1 admin-2 2000-2018; Method 1.")
    CAP["_m1_ihme_opposite"] = opp

    # ===================== Method 2 (SAE) figures =====================
    # S-1: model schematic
    fig, ax = plt.subplots(figsize=(13, 5.6)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6)

    def box(x, y, w_, h, title, body, fc, tc=NAVY):
        ax.add_patch(FancyBboxPatch((x, y), w_, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="none"))
        ax.text(x + 0.18, y + h - 0.3, title, fontsize=11, fontweight="bold", color=tc, va="top")
        ax.text(x + 0.18, y + h - 0.75, body, fontsize=9, color=INK, va="top", linespacing=1.45)

    def arr(a0, a1):
        ax.add_patch(FancyArrowPatch(a0, a1, arrowstyle="-|>", mutation_scale=16, color=NAVY, lw=1.6))
    box(0.1, 3.7, 3.7, 1.7, "1  Household surveys", "NDHS 2008, 2013, 2018, 2023-24\n148 state results with sampling error\nSets the level in every state", "#DCEBF5")
    box(0.1, 1.9, 3.7, 1.6, "2  Local context (covariates)", "Maternal care, water, wealth,\ntravel time, conflict, poverty\nfor every LGA", "#E8EEF3")
    box(0.1, 0.15, 3.7, 1.6, "3  Neighbours (spatial)", "GRID3 boundaries: 2,165 pairs of\nneighbouring LGAs share information", "#E8EEF3")
    box(4.7, 1.5, 3.9, 2.6, "Bayesian small-area model", "Each LGA has its own zero-dose rate.\nThe population-weighted average of an\nstate's LGA rates must match its survey\nresults, within sampling error.\nCovariates and neighbours explain\nhow rates differ inside each state.", "#D9EAD3", "#1C5E33")
    box(9.4, 3.2, 3.5, 2.2, "Outputs for every LGA", "Zero-dose rate and number of\nchildren, 2026-2028\n95% credible intervals\nPriority probability (top 20%)", "#FBE9DD", "#8A3A0F")
    box(9.4, 0.6, 3.5, 2.2, "Built-in checks", "Fit to all survey rounds\nIndependent survey (NmDHS 2025-26)\nIndependent LGA map (IHME 2018)\nConvergence diagnostics", "#F2F2F2")
    arr((3.8, 4.5), (4.7, 3.6)); arr((3.8, 2.7), (4.7, 2.8)); arr((3.8, 0.95), (4.7, 2.0))
    arr((8.6, 3.1), (9.4, 4.0)); arr((8.6, 2.4), (9.4, 1.7))
    ax.set_title("How the Bayesian small-area model combines surveys, local context and neighbours", loc="left", fontsize=13)
    save(fig, "M2_01_model_schematic", "Structure of the Bayesian small-area model (Method 2).", None,
         "All 774 LGAs. Inference by Markov chain Monte Carlo (4 chains x 8,000 draws).")

    # S-2: rate + CrI width maps
    g["width"] = g.m2_rate_hi95 - g.m2_rate_lo95
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    g.plot(ax=axs[0], column="m2_rate", cmap="YlOrRd", vmin=0, vmax=90, edgecolor="white", linewidth=0.08, legend=True,
           legend_kwds={"shrink": 0.6, "label": "Zero-dose rate, % of children 12-23 months"})
    basemap(axs[0], g, gs); axs[0].set_title("A. Zero-dose rate by LGA, 2026 (posterior mean)", loc="left", fontsize=11)
    g.plot(ax=axs[1], column="width", cmap="Purples", edgecolor="white", linewidth=0.08, legend=True,
           legend_kwds={"shrink": 0.6, "label": "Width of 95% credible interval (percentage points)"})
    basemap(axs[1], g, gs); axs[1].set_title("B. Uncertainty: width of the 95% credible interval", loc="left", fontsize=11)
    save(fig, "M2_02_rate_and_uncertainty_maps", "Method 2 LGA zero-dose rate for 2026 and the width of its 95% credible interval, all 774 LGAs.",
         g[["state", "lga_clean", "m2_rate", "m2_rate_lo95", "m2_rate_hi95"]],
         "Model estimate. Method 2 (Bayesian small-area estimation), 2026. n = 774 LGAs.")

    # S-3: burden + P(top 155) maps
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    g.plot(ax=axs[0], column="m2_children", cmap="Reds", edgecolor="white", linewidth=0.08, legend=True,
           legend_kwds={"shrink": 0.6, "label": "Zero-dose children, 2026"})
    basemap(axs[0], g, gs); axs[0].set_title("A. Zero-dose children by LGA, 2026", loc="left", fontsize=11)
    g.plot(ax=axs[1], column="m2_p_top155", cmap="viridis", vmin=0, vmax=1, edgecolor="white", linewidth=0.08, legend=True,
           legend_kwds={"shrink": 0.6, "label": "Probability"})
    basemap(axs[1], g, gs); axs[1].set_title("B. Probability the LGA is among the top 20% (155) by burden", loc="left", fontsize=11)
    save(fig, "M2_03_burden_and_priority_maps", "Method 2 modelled zero-dose children per LGA and the posterior probability of being among the 155 highest-burden LGAs.",
         g[["state", "lga_clean", "m2_children", "m2_p_top155"]], "Model estimate. Method 2, 2026. n = 774 LGAs.")

    # S-4: top 20 caterpillar
    top = lg.nsmallest(20, "m2_rank").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    yy = np.arange(len(top))
    ax.errorbar(top.m2_children, yy, xerr=[top.m2_children - top.m2_children_lo95, top.m2_children_hi95 - top.m2_children],
                fmt="o", color=M2C, ecolor="#C9B3A6", elinewidth=2.5, capsize=0, ms=6)
    ax.set_yticks(yy); ax.set_yticklabels([f"{l} ({s})" for l, s in zip(top.lga_clean, top.state)], fontsize=9)
    for y_, (v, p) in enumerate(zip(top.m2_children, top.m2_p_top155)):
        ax.text(top.m2_children_hi95.max() * 1.02, y_, f"P = {p:.2f}", va="center", fontsize=8.5, color=GREY)
    ax.set_xlabel("Zero-dose children, 2026 (posterior mean and 95% credible interval)"); ax.xaxis.set_major_formatter(KFMT)
    ax.set_title("Twenty highest-burden LGAs, Method 2", loc="left")
    save(fig, "M2_04_top20_lgas", "Twenty highest-burden LGAs under Method 2 with 95% credible intervals and probability of top-20% membership (P).",
         top, "Model estimate. Method 2, 2026.")

    # S-5: diagnostics panel
    O2 = ROOT / "notebooks" / "outputs_Method2_Bayesian_Small_Area_Estimation"
    ppc = pd.read_csv(O2 / "method2_sae_posterior_predictive_check.csv").rename(columns={"model": "fitted_mean", "lo95": "pi95_lo", "hi95": "pi95_hi", "inside": "in95"})
    from types import SimpleNamespace
    pr = SimpleNamespace(**R["m2"]["diagnostics"]); pr.n_params = pr.parameters
    v2 = val[(val.method == "Method 2 (SAE)") & val.check.str.startswith("NmDHS")].iloc[0]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    cols = {2008: "#B8C4CF", 2013: "#56B4E9", 2018: "#009E73", 2024: NAVY}
    for yr, cc in cols.items():
        s_ = ppc[ppc.year == yr]
        axs[0].errorbar(s_.observed, s_.fitted_mean, yerr=[s_.fitted_mean - s_.pi95_lo, s_.pi95_hi - s_.fitted_mean], fmt="o",
                        ms=3.5, color=cc, ecolor=cc, alpha=0.75, elinewidth=0.8, label=str(yr))
    axs[0].plot([0, 100], [0, 100], "--", color="#9AA5B1"); axs[0].legend(frameon=False, fontsize=8, title="NDHS round", title_fontsize=8)
    axs[0].set_xlabel("NDHS observed zero-dose, %"); axs[0].set_ylabel("Model, state aggregate (95% PI)")
    axs[0].set_title(f"A. Fit to 148 survey results: {ppc.in95.mean():.0%} in 95% PI", loc="left", fontsize=10)
    axs[1].errorbar(st.nmdhs_2025_26, st.m2_rate, yerr=[st.m2_rate - st.m2_rate_lo95, st.m2_rate_hi95 - st.m2_rate], fmt="o",
                    ms=4, color=M2C, ecolor="#E7C3AE", elinewidth=1)
    axs[1].plot([0, 90], [0, 90], "--", color="#9AA5B1")
    axs[1].set_xlabel("NmDHS 2025-26 zero-dose, % (not used in fitting)"); axs[1].set_ylabel("Model 2026, state aggregate (95% CrI)")
    axs[1].set_title(f"B. Independent survey: rho = {v2.rho:.2f}, MAE = {v2.MAE_pp:.1f} pp", loc="left", fontsize=10)
    axs[2].axis("off")
    rows = [("Chains x draws", "4 x 8,000"), ("Parameters monitored", f"{int(pr.n_params):,}"),
            ("Maximum R-hat", f"{pr.max_rhat:.3f}"), ("Minimum bulk ESS", f"{pr.min_ess_bulk:,.0f}"),
            ("Minimum tail ESS", f"{pr.min_ess_tail:,.0f}"), ("Divergent transitions", f"{int(pr.divergences)}"),
            ("Survey 95% PI coverage", f"{ppc.in95.mean():.0%}"), ("Neighbour pairs (BYM2)", "2,165")]
    tb = axs[2].table(cellText=rows, colLabels=["Diagnostic", "Value"], loc="center", cellLoc="left", colWidths=[0.65, 0.35])
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.7)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(NAVY if r_ == 0 else (LIGHT if r_ % 2 else "white"))
        if r_ == 0: cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
    axs[2].set_title("C. Convergence and calibration", loc="left", fontsize=10)
    save(fig, "M2_05_diagnostics", "Method 2 model performance: posterior predictive fit to all NDHS rounds, independent NmDHS 2025-26 check, and MCMC diagnostics.",
         ppc, "PI = posterior predictive interval including survey sampling error. R-hat <= 1.01 and no divergences indicate convergence.")

    # S-6: covariate effects
    bc = np.load(O2 / "method2_sae_lga_draws_2026.npz", allow_pickle=True)["b_cov"].astype(float)
    labels = ["Maternal care (ANC4+, facility delivery)", "Improved water", "Relative wealth", "Travel time to facility",
              "Conflict events", "Poverty rate"]
    order = np.argsort(bc.mean(0))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for k, j in enumerate(order):
        lo, hi = np.percentile(bc[:, j], [2.5, 97.5]); lo8, hi8 = np.percentile(bc[:, j], [10, 90])
        ax.plot([lo, hi], [k, k], color="#C9D3DC", lw=6, solid_capstyle="round")
        ax.plot([lo8, hi8], [k, k], color=STEEL, lw=6, solid_capstyle="round")
        ax.plot(bc[:, j].mean(), k, "o", color=NAVY, ms=8)
    ax.axvline(0, color=GREY, ls="--", lw=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[j] for j in order])
    ax.set_xlabel("Effect on zero-dose (log-odds) per standard deviation; 80% and 95% credible intervals")
    ax.set_title("What explains differences between LGAs (Method 2)", loc="left")
    save(fig, "M2_06_covariate_effects", "Posterior covariate effects in the Bayesian small-area model (log-odds of zero-dose per standard deviation).",
         pd.DataFrame({"covariate": labels, "mean": bc.mean(0), "lo95": np.percentile(bc, 2.5, 0), "hi95": np.percentile(bc, 97.5, 0)}),
         "Model estimate. Negative = lower zero-dose. Covariates: DHS spatial surfaces 2018, Meta RWI, Weiss travel time, ACLED, poverty surface.")

    # S-7: administrative coverage vs survey (why routine data alone mislead)
    inp = pd.read_csv(ROOT / "Updated Datasets for Domain 5" / "Method2_Bayesian_Small_Area_Estimation" / "sae_lga_inputs_774.csv")         .rename(columns={"admin_coverage_pct": "x", "dhis2_admin_penta1_coverage_pct": "admin_coverage_pct",
                         "dhis2_penta1_annual_mean_2023_25": "p1_annual", "lga_cohort_12_23m_2024": "N_cohort"})
    fig, (a, b2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ac = inp.admin_coverage_pct.dropna()
    a.hist(ac.clip(upper=300), bins=40, color="#9AA5B1", edgecolor="white")
    a.axvline(100, color="#C0392B", lw=2); a.text(103, a.get_ylim()[1] * 0.9, f"{(ac > 100).mean():.0%} of LGAs\nabove 100%", color="#C0392B", fontsize=10, fontweight="bold")
    a.set_xlabel("Administrative Penta1 coverage, % (DHIS2 doses / estimated cohort)"); a.set_ylabel("LGAs")
    a.set_title("A. Routine coverage exceeds 100% in most LGAs", loc="left", fontsize=10.5)
    adm = st.merge(inp.groupby("state").agg(d=("p1_annual", "sum"), n=("N_cohort", "sum")).reset_index(), on="state")
    adm["admin_zd"] = (100 - adm.d / adm.n * 100).clip(lower=0)
    b2.scatter(adm.nmdhs_2025_26, adm.admin_zd, s=34, color="#9AA5B1", edgecolor="white", label="Routine data only (100 - administrative coverage)")
    b2.scatter(adm.nmdhs_2025_26, adm.m2_rate, s=34, color=M2C, edgecolor="white", label="Method 2 (small-area estimate)")
    b2.plot([0, 85], [0, 85], "--", color="#9AA5B1"); b2.legend(frameon=False, fontsize=8.5, loc="upper left")
    from scipy.stats import spearmanr
    ra = spearmanr(adm.admin_zd, adm.nmdhs_2025_26).statistic
    b2.set_xlabel("NmDHS 2025-26 zero-dose, %"); b2.set_ylabel("Estimated zero-dose, %")
    b2.set_title(f"B. 37 states vs NmDHS: routine-only rho = {ra:.2f}; Method 2 rho = {v2.rho:.2f}", loc="left", fontsize=10.5)
    save(fig, "M2_07_admin_vs_survey", "Routine administrative coverage compared with survey-anchored small-area estimates.",
         adm[["state", "nmdhs_2025_26", "admin_zd", "m2_rate"]],
         "DHIS2 Penta1 2023-2025 annual mean over the 2024 cohort projection (state under-5 / 5, NPC 2022 shares). Sources: DHIS2; NmDHS 2025-26.")
    CAP["_admin_gt100"] = float((ac > 100).mean()); CAP["_admin_rho"] = float(ra)
    CAP["_admin_mae"] = float((adm.admin_zd - adm.nmdhs_2025_26).abs().mean())

    # ===================== Comparison figures =====================
    # C-1: zone dumbbell with intervals
    zz = zn[zn.area != "National"].sort_values("m2_children")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, r in enumerate(zz.itertuples()):
        ax.plot([r.m1_lo95, r.m1_hi95], [i + 0.15, i + 0.15], color=M1C, lw=3, alpha=0.35)
        ax.plot([r.m2_lo95, r.m2_hi95], [i - 0.15, i - 0.15], color=M2C, lw=3, alpha=0.35)
        ax.plot(r.m1_children, i + 0.15, "o", color=M1C, ms=8); ax.plot(r.m2_children, i - 0.15, "o", color=M2C, ms=8)
        ax.text(max(r.m1_hi95, r.m2_hi95) + 15000, i, f"{r.m1_children/1000:,.0f}k  vs  {r.m2_children/1000:,.0f}k", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(zz))); ax.set_yticklabels(zz.area)
    ax.set_xlabel("Zero-dose children, 2026 (estimate and 95% interval)")
    ax.legend(handles=[Line2D([], [], marker="o", color=M1C, ls="", label="Method 1"), Line2D([], [], marker="o", color=M2C, ls="", label="Method 2 (small-area)")],
              frameon=False, loc="lower right")
    ax.set_xlim(0, zz[["m1_hi95", "m2_hi95"]].max().max() * 1.28); ax.xaxis.set_major_formatter(KFMT)
    ax.set_title("Zero-dose children by geopolitical zone: two methods", loc="left")
    save(fig, "C_01_zone_comparison", "Modelled zero-dose children by zone, 2026, under Method 1 and Method 2, with 95% intervals.", zz,
         "Model estimate. Method 1 intervals reflect state-model uncertainty; Method 2 intervals are full posterior credible intervals.")

    # C-2: state comparison (37)
    ss = st.sort_values("m2_children")
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.xaxis.set_major_formatter(KFMT); ax.tick_params(axis="x", labelsize=11)
    for i, r in enumerate(ss.itertuples()):
        ax.plot([r.m2_children_lo95, r.m2_children_hi95], [i - 0.17, i - 0.17], color=M2C, lw=2.2, alpha=0.35)
        ax.plot([r.m1_children_lo95, r.m1_children_hi95], [i + 0.17, i + 0.17], color=M1C, lw=2.2, alpha=0.35)
        ax.plot(r.m1_children, i + 0.17, "o", color=M1C, ms=5.5); ax.plot(r.m2_children, i - 0.17, "o", color=M2C, ms=5.5)
    ax.set_yticks(range(len(ss))); ax.set_yticklabels(ss.state, fontsize=11); ax.set_ylim(-0.8, len(ss) - 0.2)
    ax.set_xlabel("Zero-dose children, 2026 (estimate and 95% interval)")
    ax.legend(handles=[Line2D([], [], marker="o", color=M1C, ls="", label="Method 1"), Line2D([], [], marker="o", color=M2C, ls="", label="Method 2 (small-area)")],
              frameon=False, loc="lower right")
    ax.set_title("Zero-dose children by state: two methods", loc="left")
    save(fig, "C_02_state_comparison", "Modelled zero-dose children by state, 2026, under both methods.", ss, "Model estimate. 37 states including FCT.")

    # C-3: agreement with independent data (Spearman with 95% CI)
    vv = val[val.method.isin(["Method 1", "Method 2 (SAE)"])].copy()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    checks = [("NmDHS 2025-26, 37 states (state 2026 estimate vs survey)", "State ranking vs NmDHS 2025-26 (37 states)"),
              ("IHME DTP1 2018, LGA overall", f"LGA ranking vs IHME 2018 ({int(vv[vv.check == 'IHME DTP1 2018, LGA overall'].n.iloc[0])} LGAs)"),
              ("IHME DTP1 2018, LGA within state (pooled within-state ranks)", "LGA ranking within states vs IHME 2018")]
    for i, (ck, lab) in enumerate(checks):
        for m, col, off in [("Method 1", M1C, 0.16), ("Method 2 (SAE)", M2C, -0.16)]:
            r = vv[(vv.method == m) & (vv.check == ck)].iloc[0]
            ax.plot([r.ci_lo, r.ci_hi], [i + off] * 2, color=col, lw=3, alpha=0.4)
            ax.plot(r.rho, i + off, "o", color=col, ms=9)
            ax.text(r.ci_hi + 0.02, i + off, f"{r.rho:.2f}" + (" (p<0.001)" if r.p_value < 0.001 else f" (p={r.p_value:.3f})"),
                    va="center", fontsize=9, color=col)
    ax.set_yticks(range(3)); ax.set_yticklabels([c[1] for c in checks]); ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlim(-0.1, 1.15); ax.set_xlabel("Spearman rank correlation (95% bootstrap CI)")
    ax.legend(handles=[Line2D([], [], marker="o", color=M1C, ls="", label="Method 1"), Line2D([], [], marker="o", color=M2C, ls="", label="Method 2 (small-area)")],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.invert_yaxis(); ax.set_title("Agreement with independent data", loc="left")
    save(fig, "C_03_agreement_independent_data", "Spearman rank correlation of each method with independent data: NmDHS 2025-26 (states) and IHME 2018 (LGAs, overall and within state).",
         vv, "IHME 2018 is eight years earlier than the 2026 estimates; it tests the geographic pattern, not 2026 levels.")

    # C-4: Pareto both methods
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.fill_between(curves.k, curves.m2_lo95, curves.m2_hi95, color=M2C, alpha=0.15, lw=0)
    ax.plot(curves.k, curves.m2_cum_pct, color=M2C, lw=2.2, label="Method 2 (small-area), 774 LGAs")
    ax.plot(curves.k, curves.m1_cum_pct, color=M1C, lw=2.2, label="Method 1, 773 LGAs")
    for _, r in pareto.iterrows():
        s_ = int(r.share_of_burden); ax.axhline(s_, color="#C9D3DC", lw=0.8, ls=":")
        ax.plot(r.m1_lgas, s_, "o", color=M1C); ax.plot(r.m2_lgas, s_, "o", color=M2C)
        ax.annotate(f"{s_}%: {int(r.m2_lgas)} vs {int(r.m1_lgas)} LGAs", (r.m1_lgas, s_), xytext=(12, -12), textcoords="offset points", fontsize=9.5, color=INK)
    ax.set_xlim(0, 774); ax.set_ylim(0, 101); ax.set_xlabel("Number of LGAs, ranked by modelled burden"); ax.set_ylabel("Cumulative share of zero-dose children (%)")
    ax.legend(frameon=False, loc="lower right"); ax.set_title("How many LGAs hold 50%, 60% and 80% of zero-dose children", loc="left")
    save(fig, "C_04_pareto_both_methods", "Cumulative share of modelled zero-dose children by number of LGAs targeted, both methods (Method 2 band = 95% credible interval).",
         curves, "Model estimate, 2026.")

    # C-5: LGA scatter
    d = lg.dropna(subset=["m1_children"])
    fig, (a, b2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    a.scatter(d.m1_rate, d.m2_rate, s=9, alpha=0.5, color=NAVY, edgecolor="none"); a.plot([0, 100], [0, 100], "--", color="#9AA5B1")
    a.set_xlabel("Method 1 zero-dose rate, %"); a.set_ylabel("Method 2 zero-dose rate, %"); a.set_title("A. LGA zero-dose rate", loc="left")
    a.text(0.03, 0.93, f"Spearman rho = {spearmanr(d.m1_rate, d.m2_rate).statistic:.2f}", transform=a.transAxes)
    b2.scatter(d.m1_children, d.m2_children, s=9, alpha=0.5, color=NAVY, edgecolor="none"); mx = max(d.m1_children.max(), d.m2_children.max())
    b2.plot([0, mx], [0, mx], "--", color="#9AA5B1"); b2.set_xlabel("Method 1 zero-dose children"); b2.set_ylabel("Method 2 zero-dose children")
    b2.set_title("B. LGA zero-dose children", loc="left")
    b2.text(0.03, 0.93, f"Spearman rho = {spearmanr(d.m1_children, d.m2_children).statistic:.2f}", transform=b2.transAxes)
    save(fig, "C_05_lga_scatter", "LGA-level agreement between Method 1 and Method 2, 773 LGAs.", d[["state", "lga_clean", "m1_rate", "m2_rate", "m1_children", "m2_children"]],
         "Model estimate, 2026.")

    # ===================== Archetypes =====================
    cov = pd.read_parquet(C.D["integ"] / "domain5_all774_covariates_and_missingness.parquet")[["state", "lga_clean", "anc4plus", "delivery_hf"]]
    ga = g.copy()
    fig = plt.figure(figsize=(14, 7))
    ax0 = fig.add_axes([0.0, 0.2, 0.47, 0.75])
    for k in range(1, 6):
        sub = ga[ga.archetype == k]
        sub.plot(ax=ax0, color=ARC[k], edgecolor="white", linewidth=0.08)
    basemap(ax0, ga, gs)
    ax0.legend(handles=[Patch(color=ARC[k], label=f"{k}  {ARCN[k]} ({int((lg.archetype == k).sum())})") for k in range(1, 6)],
               loc="upper left", fontsize=9, frameon=False, bbox_to_anchor=(0.0, 0.02))
    ax0.set_title("A. Five structural archetypes, 774 LGAs", loc="left", fontsize=11)
    ax1 = fig.add_axes([0.55, 0.25, 0.43, 0.65])
    x = np.arange(5); w = 0.27
    ax1.bar(x - w, arch.cohort_share, w, color="#C9D3DC", label="Share of children 12-23 months")
    ax1.bar(x, arch.m1_share, w, color=M1C, label="Share of zero-dose, Method 1")
    ax1.bar(x + w, arch.m2_share, w, color=M2C, label="Share of zero-dose, Method 2")
    for xi, (a1, a2) in enumerate(zip(arch.m1_children, arch.m2_children)):
        ax1.text(xi, max(arch.m1_share[xi], arch.m2_share[xi], arch.cohort_share[xi]) + 1.2, f"{a1/1e3:,.0f}k | {a2/1e3:,.0f}k",
                 ha="center", fontsize=8.5, color=INK)
    ax1.set_xticks(x); ax1.set_xticklabels([f"A{k}" for k in range(1, 6)]); ax1.set_ylabel("%")
    ax1.legend(frameon=False, fontsize=8.5); ax1.set_title("B. Burden is concentrated beyond population share", loc="left", fontsize=11)
    save(fig, "A_01_archetype_map_burden", "LGA structural archetypes (Ward clustering of 15 covariates) and the share of modelled zero-dose children in each, under both methods.",
         arch, "Model estimate, 2026. Labels above bars: Method 1 | Method 2 zero-dose children.")

    # A-2: ANC vs rate scatter + maternal-care bars by archetype
    dd = lg.merge(cov, on=["state", "lga_clean"], how="left")
    fig, (a, b2) = plt.subplots(1, 2, figsize=(13, 5))
    for k in range(1, 6):
        s_ = dd[dd.archetype == k]
        a.scatter(s_.anc4plus, s_.m1_rate, s=10, alpha=0.6, color=ARC[k], label=f"A{k}")
    v = dd.dropna(subset=["anc4plus"])
    r1 = np.corrcoef(v.dropna(subset=["m1_rate"]).anc4plus, v.dropna(subset=["m1_rate"]).m1_rate)[0, 1]
    r2 = np.corrcoef(v.anc4plus, v.m2_rate)[0, 1]
    a.set_xlabel("Antenatal care, four or more visits (%)"); a.set_ylabel("Zero-dose rate 2026, % (Method 1)")
    a.set_title(f"A. ANC4+ vs zero-dose rate: r = {r1:.2f} (Method 1, {int(v.m1_rate.notna().sum())} LGAs)", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=8, markerscale=2)
    mc = dd.groupby("archetype").agg(anc=("anc4plus", "mean"), deliv=("delivery_hf", "mean"), zd1=("m1_rate", "mean"), zd2=("m2_rate", "mean")).reset_index()
    x = np.arange(5); w = 0.2
    for off, col, c_, lab in [(-1.5, "anc", "#7FA7C9", "ANC4+"), (-0.5, "deliv", "#2E6E8E", "Facility delivery"),
                              (0.5, "zd1", M1C, "Zero-dose, Method 1"), (1.5, "zd2", M2C, "Zero-dose, Method 2")]:
        b2.bar(x + off * w, mc[col], w, color=c_, label=lab)
    b2.set_xticks(x); b2.set_xticklabels([f"A{k}" for k in mc.archetype]); b2.set_ylabel("%"); b2.legend(frameon=False, fontsize=8.5)
    b2.set_title("B. Maternal care and zero-dose by archetype", loc="left", fontsize=10)
    save(fig, "A_02_maternal_care", "Antenatal care and facility delivery against modelled zero-dose by archetype.", mc,
         f"Model estimate. n = {len(v)} LGAs with maternal-care data ({int(v.m1_rate.notna().sum())} with a Method 1 estimate). DHS Spatial Data Repository 2018 surfaces.")
    CAP["_anc_r_m1"], CAP["_anc_r_m2"], CAP["_anc_n"] = float(r1), float(r2), int(len(v))
    r1d = np.corrcoef(v.dropna(subset=["m1_rate", "delivery_hf"]).delivery_hf, v.dropna(subset=["m1_rate", "delivery_hf"]).m1_rate)[0, 1]
    r2d = np.corrcoef(v.dropna(subset=["delivery_hf"]).delivery_hf, v.dropna(subset=["delivery_hf"]).m2_rate)[0, 1]
    CAP["_deliv_r_m1"], CAP["_deliv_r_m2"] = float(r1d), float(r2d)
    CAP["_mc"] = mc.round(1).to_dict(orient="records")

    (FIG / "captions.json").write_text(json.dumps(CAP, indent=2, default=float))
    print(json.dumps({k: v for k, v in CAP.items() if k.startswith("_")}, indent=1, default=float))
    print(sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
