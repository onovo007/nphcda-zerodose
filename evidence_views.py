"""
Validation and contextual-profile evidence views (bundled project data).

Two blocks, both drawn from small text files in data/sample/two_methods/evidence/ that hold the verified
refinement outputs:
- render_sae_evidence: comparative accuracy against the NmDHS 2025-26, calibration of the uncertainty
  intervals, survey-round holdouts and burden capture of priority lists (Method 2 and comparators).
- render_profile_evidence: dominant contextual barrier and profile membership by LGA, profile domain
  scores, within-state variation, agreement of package-assignment rules and package stability.
All figures are model estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ai
import config as C
import viz
from theme import clean, kpi_row, section, style_fig

EV_DIR = C.DATA_DIR / "two_methods" / "evidence"
DOMS = ["Socioeconomic", "Maternal care", "Nutrition", "Access", "Insecurity", "Programme (dropout)"]
DOM_LABEL = {d: ("DTP1-3 dropout" if d.startswith("Programme") else d) for d in DOMS}
DOM_COLOR = {"Socioeconomic": "#8C564B", "Maternal care": "#CC79A7", "Nutrition": "#D55E00", "Access": "#0072B2",
             "Insecurity": "#000000", "Programme (dropout)": "#E69F00", "None above +0.5 SD": "#E5E5E5"}
MEMB_COLOR = {"Clear": "#2E7D32", "Intermediate": "#F2C14E", "Mixed": "#C0392B"}
PROFILES = {1: "Deprived northern rural", 2: "Low maternal care, high conflict exposure", 3: "Geographically isolated",
            4: "Near-average", 5: "Relatively advantaged"}
APPROACH_COLOR = {"Method 2 (SAE)": C.NPHCDA_GREEN, "Method 1 (hierarchical state model)": C.NAVY,
                  "Gradient boosting": "#7B5EA7", "Fay-Herriot area-level model": C.GOLD,
                  "Latest survey carried forward": "#8C9AAB", "Routine data only": C.ACCENT}
RULES = {"Routine administrative proxy": "Routine administrative coverage", "Prevalence only": "Method 2 rate only",
         "Burden, state model": "Method 1 expected burden", "Burden, SAE": "Method 2 expected burden",
         "Uncertainty-aware: probability of being in the top 155": "Method 2 probability of top-155 membership",
         "Oracle": "Upper bound (true top-K in each draw)"}


def available() -> bool:
    return (EV_DIR / "validation_comparators.csv").exists() and (EV_DIR / "lga_barriers_774.csv").exists()


@st.cache_data(show_spinner=False)
def load() -> dict:
    return {f.stem: pd.read_csv(f) for f in EV_DIR.glob("*.csv")}


def lga_barriers() -> pd.DataFrame | None:
    """Per-LGA profile, membership, dominant barrier, flagged barriers and candidate components."""
    return load().get("lga_barriers_774") if available() else None


# --------------------------------------------------------------------------- Method 2 evidence
def _accuracy_fig(v: pd.DataFrame) -> go.Figure:
    a = v[v["check"] == "NmDHS 2025-26"].sort_values("MAE", ascending=False)
    d = a[a["approach"] != "Method 2 (SAE)"].dropna(subset=["dMAE_vs_SAE"]).sort_values("dMAE_vs_SAE")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.22,
                        subplot_titles=("State-level error (MAE, points)", "Extra error over Method 2 (95% CI)"))
    fig.add_bar(y=a["approach"], x=a["MAE"], orientation="h", marker_color=[APPROACH_COLOR[x] for x in a["approach"]],
                text=[f"{x:.1f}" for x in a["MAE"]], textposition="outside", showlegend=False, row=1, col=1)
    d = d[d["approach"] != "Routine data only"]
    fig.add_scatter(y=d["approach"], x=d["dMAE_vs_SAE"], mode="markers", marker=dict(size=10, color=C.NAVY),
                    error_x=dict(type="data", symmetric=False, array=d["dMAE_hi"] - d["dMAE_vs_SAE"],
                                 arrayminus=d["dMAE_vs_SAE"] - d["dMAE_lo"]), showlegend=False, row=1, col=2)
    fig.add_vline(x=0, line_dash="dot", line_color=C.ACCENT, row=1, col=2)
    fig.update_xaxes(range=[0, 31], row=1, col=1)
    return style_fig(fig, height=380)


def _holdout_fig(v: pd.DataFrame) -> go.Figure:
    h = v[v["check"].str.startswith("Predict")]
    fig = go.Figure()
    for ap in [x for x in APPROACH_COLOR if x != "Routine data only"]:
        q = h[h["approach"] == ap].sort_values("check")
        fig.add_bar(x=q["check"], y=q["MAE"], name=ap, marker_color=APPROACH_COLOR[ap])
    fig.update_layout(barmode="group", title="Survey-round holdouts: error when a round is predicted from earlier rounds",
                      yaxis_title="MAE (percentage points)", legend=dict(orientation="h", y=-0.2))
    return style_fig(fig, height=400)


def _calibration_fig(cal: pd.DataFrame) -> go.Figure:
    keep = cal["interval"].str.contains("latent") | cal["interval"].str.contains("design effect 2")
    fig = go.Figure()
    fig.add_scatter(x=[0.45, 1], y=[0.45, 1], mode="lines", line=dict(dash="dot", color="#999"), name="Perfect calibration")
    for (m, i), q in cal[keep].groupby(["model", "interval"]):
        lab = f"{'Method 2' if 'SAE' in m else 'Method 1'}, {'credible (rate)' if 'latent' in i else 'predictive (survey)'}"
        fig.add_scatter(x=q["nominal"], y=q["empirical_coverage"], mode="lines+markers", name=lab,
                        line=dict(color=C.NPHCDA_GREEN if "SAE" in m else C.NAVY, dash="solid" if "predictive" in i else "dash"))
    fig.update_layout(title="Calibration: share of the 37 NmDHS state values inside each interval",
                      xaxis_title="Nominal coverage", yaxis_title="Empirical coverage",
                      xaxis=dict(tickformat=".0%"), yaxis=dict(tickformat=".0%"), legend=dict(orientation="h", y=-0.25))
    return style_fig(fig, height=430)


def _temporal_fig(t: pd.DataFrame) -> go.Figure:
    t = t.copy()
    t["Projected direction"] = np.where(t["direction_reversal"], "Wrong direction", "Right direction")
    t["Holdout"] = "Predict " + t["holdout"].astype(str)
    fig = px.scatter(t, x="observed_change", y="model_implied_change", color="Projected direction", facet_col="Holdout",
                     hover_name="state", color_discrete_map={"Wrong direction": C.ACCENT, "Right direction": C.NAVY},
                     labels={"observed_change": "Observed change from last training round (points)",
                             "model_implied_change": "Model-implied change (points)"})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.add_hline(y=0, line_color="#bbb"); fig.add_vline(x=0, line_color="#bbb")
    fig.update_layout(title="Temporal extrapolation by state", legend=dict(orientation="h", y=-0.25))
    return style_fig(fig, height=430)


def _capture_fig(c: pd.DataFrame) -> go.Figure:
    c = c.copy()
    c["Rule"] = c["rule"].map(lambda r: next((v for k, v in RULES.items() if r.startswith(k)), None))
    c = c.dropna(subset=["Rule"])
    col = {"Routine administrative coverage": C.ACCENT, "Method 2 rate only": C.GOLD, "Method 1 expected burden": C.NAVY,
           "Method 2 expected burden": C.NPHCDA_GREEN, "Method 2 probability of top-155 membership": "#7B5EA7",
           "Upper bound (true top-K in each draw)": "#9AA5B1"}
    fig = go.Figure()
    for r, q in c.groupby("Rule"):
        fig.add_scatter(x=q["K"], y=q["capture_SAE_mean"], mode="lines+markers", name=r,
                        line=dict(color=col[r], dash="dot" if r.startswith("Upper") else "solid"))
    fig.add_vline(x=155, line_dash="dot", line_color="#999")
    fig.update_layout(title="Share of modelled zero-dose children captured by priority lists (scored under Method 2)",
                      xaxis_title="Number of LGAs selected (programme capacity)", yaxis_title="Captured (%)",
                      legend=dict(orientation="h", y=-0.25))
    return style_fig(fig, height=450)


def render_sae_evidence(tm: dict):
    """Comparative accuracy, calibration, temporal holdouts and uncertainty-aware prioritization."""
    if not available():
        return
    E = load()
    v, cal = E["validation_comparators"], E["calibration"]
    lg = tm["lga"]
    p = lg["m2_p_top155"]
    near, unc, unl = int((p >= 0.9).sum()), int(((p > 0.1) & (p < 0.9)).sum()), int((p <= 0.1).sum())
    unc_off = int((((p > 0.1) & (p < 0.9)) & (lg["m2_rank"] > 155)).sum())
    sae = v[(v["check"] == "NmDHS 2025-26") & (v["approach"] == "Method 2 (SAE)")].iloc[0]
    section("How accurate and how honest are the estimates?",
            "Method 2 compared with four alternative approaches against the NmDHS 2025-26, which no model "
            "used in fitting; calibration of the uncertainty intervals; and survey-round holdouts.")
    kpi_row([
        {"label": "Method 2 error vs NmDHS", "value": f"MAE {sae['MAE']:.1f}",
         "sub": clean(f"points; bias {sae['bias']:+.1f}; lowest of six approaches"), "color": C.NPHCDA_GREEN},
        {"label": "Rank agreement", "value": f"rho {sae['spearman_rho']:.2f}",
         "sub": clean(f"95% CI {sae['spearman_ci_lo']:.2f} to {sae['spearman_ci_hi']:.2f}"), "color": C.NAVY},
        {"label": "95% predictive interval", "value": "37 of 37",
         "sub": "states covered (50% interval: 24 of 37)", "color": C.STEEL},
        {"label": "Near-certain priorities", "value": f"{near} LGAs",
         "sub": clean(f"P(top 155) of 0.9 or more; {unc} uncertain, {unc_off} of them outside the list"),
         "color": C.GOLD},
    ])
    t1, t2, t3, t4 = st.tabs(["Comparative accuracy", "Calibration", "Holdouts and trend breaks", "Prioritizing under uncertainty"])
    with t1:
        st.plotly_chart(_accuracy_fig(v), use_container_width=True)
        tb = v[v["check"] == "NmDHS 2025-26"].sort_values("MAE")
        st.dataframe(pd.DataFrame({"Approach": tb["approach"], "MAE (points)": tb["MAE"].round(1),
                                   "Bias (points)": tb["bias"].round(1), "Spearman rho": tb["spearman_rho"].round(2),
                                   "CRPS": tb["CRPS"].where(~tb["approach"].isin(
                                       ["Gradient boosting", "Routine data only"])).round(1),
                                   "Extra MAE over Method 2 (95% CI)": [
                                       "Reference" if a == "Method 2 (SAE)" else f"{d:+.1f} ({lo:.1f} to {hi:.1f})"
                                       for a, d, lo, hi in zip(tb["approach"], tb["dMAE_vs_SAE"], tb["dMAE_lo"], tb["dMAE_hi"])]}),
                     use_container_width=True)
        st.caption(clean("MAE = mean absolute error in percentage points; CRPS = continuous ranked probability score "
                         "(lower is better); CI = bootstrap confidence interval over the 37 states. Method 1 and "
                         "Method 2 are statistically equivalent at state level; routine administrative data alone "
                         "are far less accurate."))
    with t2:
        st.plotly_chart(_calibration_fig(cal), use_container_width=True)
        st.caption(clean("Predictive intervals add survey sampling error (design effect 2) to the uncertainty in the "
                         "rate; they are the right comparison for a single survey value. Points on the dotted line "
                         "mean the intervals cover as often as they claim."))
    with t3:
        st.plotly_chart(_holdout_fig(v), use_container_width=True)
        st.plotly_chart(_temporal_fig(E["temporal_holdout"]), use_container_width=True)
        st.caption(clean("Method 2 predicted 2018 better than carrying the 2013 survey forward but predicted 2024 less "
                         "well than carrying 2018 forward: between 2018 and 2024 many states changed direction (red). "
                         "Refresh the model whenever a new survey round becomes available."))
    with t4:
        st.plotly_chart(_capture_fig(E["capture_curves"]), use_container_width=True)
        st.caption(clean(f"Membership of the 155 highest-burden LGAs across 4,000 posterior draws: {near} near-certain "
                         f"(0.9 or more), {unc} uncertain and {unl} unlikely (0.1 or less). Fund near-certain LGAs "
                         "first and verify uncertain ones through microplanning. Capture shares are model-based, "
                         "not observed outcomes."))
    ai.ai_block("d5_sae_evidence", "Zero-Dose & Hotspots - accuracy, calibration and prioritization evidence",
                "Accuracy of Method 2 and alternatives against the NmDHS 2025-26, interval calibration, survey-round "
                "holdouts and priority-list capture. Say what can and cannot be relied on for targeting.",
                {"accuracy": v.round(2).to_dict(orient="records"),
                 "priority_membership": {"near_certain": near, "uncertain": unc, "uncertain_outside_list": unc_off,
                                         "unlikely": unl}})


# --------------------------------------------------------------------------- profile evidence
def _lga_map(tm: dict, b: pd.DataFrame, col: str, cmap: dict, title: str, legend: str) -> go.Figure:
    import spatial
    k = tm["lga"][["lga_uid", "state_key", "geo_lga_key"]].merge(b[["lga_uid", col]], on="lga_uid")
    g = spatial.load_gdf("lga").merge(k.drop(columns="lga_uid").rename(columns={"geo_lga_key": "lga_key"}),
                                      on=["state_key", "lga_key"], how="left")
    g[col] = g[col].fillna("Not classified")
    return viz.choropleth(g, col, categorical=True, color_map={**cmap, "Not classified": "#FFFFFF"}, title=title,
                          legend_title=legend, height=560)


def _heatmap(b: pd.DataFrame) -> go.Figure:
    m = b.groupby("profile_id")[[f"score_{d}" for d in DOMS]].mean()
    n = b.groupby("profile_id").size()
    fig = go.Figure(go.Heatmap(z=m.values, x=[DOM_LABEL[d] for d in DOMS],
                               y=[f"{PROFILES[i]} ({n[i]})" for i in m.index], colorscale="RdBu_r", zmid=0,
                               zmin=-2, zmax=2, text=np.round(m.values, 2), texttemplate="%{text:+.2f}",
                               colorbar=dict(title="SD")))
    fig.update_layout(title="Mean domain score by contextual profile (higher = more disadvantaged)",
                      yaxis=dict(autorange="reversed", automargin=True))
    return style_fig(fig, height=380)


def _variance_rules_fig(vd: pd.DataFrame, rr: pd.DataFrame) -> go.Figure:
    vd = vd.sort_values("share_variance_within_states")
    rr = rr.sort_values("burden_weighted_jaccard")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.25,
                        subplot_titles=("Share of variance within states (%)", "Agreement with LGA barrier packages"))
    fig.add_bar(y=[DOM_LABEL[d] for d in vd["domain"]], x=vd["share_variance_within_states"] * 100, orientation="h",
                marker_color=[DOM_COLOR[d] for d in vd["domain"]], text=[f"{x:.0f}%" for x in vd["share_variance_within_states"] * 100],
                textposition="outside", showlegend=False, row=1, col=1)
    fig.add_bar(y=rr["assignment_rule"], x=rr["burden_weighted_jaccard"], orientation="h", marker_color=C.NAVY,
                text=[f"{x:.2f}" for x in rr["burden_weighted_jaccard"]], textposition="outside", showlegend=False,
                row=1, col=2)
    fig.update_xaxes(range=[0, 75], row=1, col=1)
    fig.update_xaxes(range=[0, 1], row=1, col=2)
    return style_fig(fig, height=400)


def _stability_fig(s: pd.DataFrame) -> go.Figure:
    s = s[~s["alternative"].str.startswith("LGA barrier")].iloc[::-1]
    lab = (s["alternative"].str.replace("Ward original, k=", "Original indicators, k=", regex=False)
           .str.replace("Ward domain_balanced_no_dtp, k=", "Domain-balanced without dropout, k=", regex=False)
           .str.replace("Ward domain_balanced, k=", "Domain-balanced, k=", regex=False)
           .str.replace("Ward no_dtp_dropout, k=", "Without DTP1-3 dropout, k=", regex=False))
    fig = go.Figure()
    for c, name, colr in [("same_primary_profile", "Same primary profile", "#9AA5B1"),
                          ("same_core_lgas", "Same core components (LGAs)", C.GOLD),
                          ("same_core_children", "Same core components (modelled children)", C.NAVY)]:
        fig.add_bar(y=lab, x=s[c], name=name, orientation="h", marker_color=colr)
    fig.update_layout(barmode="group", title="Stability of candidate packages across modelling choices",
                      xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis=dict(automargin=True),
                      legend=dict(orientation="h", y=-0.15))
    return style_fig(fig, height=620)


def render_profile_evidence(tm: dict):
    """Contextual barriers, profile robustness and package-assignment evidence."""
    if not available():
        return
    E = load()
    b = E["lga_barriers_774"]
    m2 = tm["lga"][["lga_uid", "m2_children"]]
    bb = b.merge(m2, on="lga_uid", how="left")
    multi = bb["n_barriers"] >= 2
    st_multi = int((b.groupby("state")["dominant_barrier"].nunique() >= 2).sum())
    vd = E["within_state_variance"].set_index("domain")["share_variance_within_states"] * 100
    rr = E["assignment_rules"].set_index("assignment_rule")["burden_weighted_jaccard"]
    section("Contextual barriers and candidate packages",
            "What drives zero-dose differs between LGAs, even inside one state and at the same estimated rate. "
            "Profile labels describe only what the 15 indicators measure.")
    kpi_row([
        {"label": "Two or more flagged barriers", "value": f"{multi.mean() * 100:.0f}% of LGAs",
         "sub": clean(f"holding {bb.loc[multi, 'm2_children'].sum() / bb['m2_children'].sum() * 100:.0f}% of modelled zero-dose children"),
         "color": C.ACCENT},
        {"label": "States with mixed barriers", "value": f"{st_multi} of 37",
         "sub": "contain LGAs with different dominant barriers", "color": C.NAVY},
        {"label": "Variance within states", "value": f"{vd['Access']:.0f}% / {vd['Insecurity']:.0f}%",
         "sub": clean(f"access / insecurity (nutrition {vd['Nutrition']:.0f}%)"), "color": C.GOLD},
        {"label": "Best assignment rule", "value": f"State {rr['By state']:.2f}",
         "sub": clean(f"then profile {rr['By contextual profile']:.2f}; one national package {rr['One national package']:.2f}"),
         "color": C.STEEL,
         "help": "Burden-weighted Jaccard agreement with each LGA's own barrier package (1 = identical). "
                 "Agreement with documented rules, not measured intervention effects."},
    ])
    t1, t2, t3, t4 = st.tabs(["Dominant barrier map", "Profile signatures", "Where packages should be set",
                              "Robustness and rules"])
    with t1:
        st.plotly_chart(_lga_map(tm, b, "dominant_barrier", DOM_COLOR,
                                 "Dominant contextual barrier (domain score of +0.5 SD or more)", "Barrier"),
                        use_container_width=True)
        st.plotly_chart(_lga_map(tm, b, "membership", MEMB_COLOR, "Profile membership: clear, intermediate or mixed",
                                 "Membership"), use_container_width=True)
        st.caption(clean("Clear: bootstrap assignment frequency of 0.8 or more and centroid distance ratio below 0.8; "
                         "mixed: frequency below 0.5 or ratio of 0.9 or more. These are stability and affinity "
                         "measures, not probabilities."))
    with t2:
        st.plotly_chart(_heatmap(b), use_container_width=True)
        st.caption(clean("Five profiles are an operational choice: four separate the data slightly more cleanly "
                         "(silhouette 0.31 vs 0.18) by merging near-average and relatively advantaged LGAs, which "
                         "differ in burden. 79% of neighbouring LGAs share a profile (26% by chance). Earlier labels "
                         "naming nomadic, migrant, riverine and urban-slum areas were retired: no indicator measures "
                         "those attributes."))
    with t3:
        st.plotly_chart(_variance_rules_fig(E["within_state_variance"], E["assignment_rules"]), use_container_width=True)
        st.caption(clean("State packages refined with each LGA's flagged barriers agree best with the data; a single "
                         "national package agrees least. Profiles add to zone in explaining LGA zero-dose but little "
                         "beyond a deprivation quintile, so use them as a planning summary."))
    with t4:
        st.plotly_chart(_stability_fig(E["package_stability"]), use_container_width=True)
        mat = E["barrier_matrix"].rename(columns={"barrier": "Barrier", "indicator": "Indicators", "rule": "Flagging rule",
                                                  "components": "Candidate components", "evidence": "Evidence or guideline",
                                                  "local_validation": "Local validation required"})
        st.markdown("**Barrier-to-intervention mapping**")
        st.dataframe(mat, use_container_width=True, hide_index=True)
        st.caption(clean("Core components (Jaccard of 0.67 or more) are kept for 51-82% of LGAs (71-96% of modelled "
                         "children) across alternatives. A common core of fixed-site and outreach routine "
                         "immunization with microplanning applies to every LGA. Validate components locally."))
    ai.ai_block("profile_evidence", "LGA Priority & Archetypes - contextual barriers and candidate packages",
                "Dominant barriers by LGA, within-state variation, agreement of package-assignment rules and package "
                "stability. Explain how a planner should choose packages and what needs local validation.",
                {"within_state_variance_pct": vd.round(0).to_dict(), "assignment_rule_agreement": rr.round(2).to_dict(),
                 "share_lgas_two_plus_barriers": round(float(multi.mean() * 100), 1),
                 "dominant_barrier_counts": b["dominant_barrier"].value_counts().to_dict()})
