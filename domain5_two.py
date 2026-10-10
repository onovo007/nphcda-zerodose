"""Domain 5 view for the bundled project data - zero-dose estimates from two Bayesian methods."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
import names as N
import viz
import ai
from theme import section, kpi_row, clean, domain_banner, style_fig, highlight_classes
from models import d5_two_methods as TM

BAND_CELL = {
    "A: first 50%": "background-color:#FDE2E0;color:#7F1D1D;font-weight:700",
    "B: 50-60%": "background-color:#FDEBD9;color:#7A3E00;font-weight:600",
    "C: 60-80%": "background-color:#FEF6D6;color:#6B5300",
    "D: 80-100%": "background-color:#E6F0F6;color:#1F3B57",
}
LIST_COLS = ["Burden rank", "State", "LGA", "Zone", "Zero-dose children (est)", "Children low (95%)",
             "Children high (95%)", "Zero-dose rate (%)", "Rate low (95%)", "Rate high (95%)",
             "Cumulative % of burden", "Priority band", "Archetype"]


def _download(df, label, fname, key=None):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), fname, "text/csv", key=key)


def _m(v):
    return f"{v / 1e6:.2f}M"


def method_text(method: str) -> str:
    if method == TM.M1:
        return clean(
            "State zero-dose rates come from a Bayesian hierarchical Beta regression on the NDHS "
            "2008-2024 panel (partially pooled national, zone and state intercepts and time slopes, "
            "with the state DHIS2 Penta1 trend as a covariate), forecast to 2026-2028. Each state's "
            "2026 burden is then allocated to its LGAs using the LGA's DHIS2 Penta1 share, calibrated to "
            "the state posterior, and its NPC 2022 population share; the state credible interval is "
            "carried down to each LGA. Covers 773 LGAs (Guzamala, Borno has no Penta1 data for "
            "2021-2024).")
    return clean(
        "An LGA-level Bayesian model fitted directly to the NDHS 2008-2024 state survey observations "
        "through an aggregation likelihood (the population-weighted mean of a state's LGA rates must agree "
        "with its survey result, allowing for sampling error with design effect 2). Differences between "
        "LGAs inside a state come from six local-condition covariates (maternal-care index, improved water "
        "source, Relative Wealth Index, travel time to the nearest facility, conflict events 2021-2024, "
        "poverty rate) and a BYM2 spatial effect on the GRID3 LGA adjacency. Every one of the 774 LGAs "
        "receives its own posterior rate and 95% credible interval.")


def _pareto_fig(ranked: pd.DataFrame, title: str) -> go.Figure:
    x = ranked["Burden rank"].values
    cum = ranked["Cumulative % of burden"].values
    band_col = {"A: first 50%": C.ACCENT, "B: 50-60%": "#E07B39", "C: 60-80%": C.GOLD, "D: 80-100%": C.STEEL}
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=ranked["Zero-dose children (est)"],
                         marker_color=[band_col[b] for b in ranked["Priority band"]], name="Children per LGA",
                         customdata=np.stack([ranked["LGA"], ranked["State"]], axis=1),
                         hovertemplate="Rank %{x}: %{customdata[0]} (%{customdata[1]})<br>"
                                       "%{y:,.0f} children<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=cum, yaxis="y2", mode="lines", name="Cumulative % of burden",
                             line=dict(color=C.NAVY, width=2.4),
                             hovertemplate="Top %{x} LGAs: %{y:.1f}%<extra></extra>"))
    for share in TM.SHARES:
        n = TM.n_for_share(ranked, share)
        fig.add_vline(x=n, line=dict(color="#475569", width=1, dash="dot"))
        fig.add_annotation(x=n, y=share, yref="y2", text=f"{share}%: {n} LGAs", showarrow=True,
                           arrowhead=2, ax=55, ay=-25, font=dict(size=12, color=C.NAVY),
                           bgcolor="rgba(255,255,255,0.85)")
    fig.update_layout(title=title, xaxis_title="LGAs ranked by estimated zero-dose children",
                      yaxis=dict(title="Zero-dose children per LGA"),
                      yaxis2=dict(title="Cumulative % of national burden", overlaying="y", side="right",
                                  range=[0, 100], showgrid=False),
                      legend=dict(orientation="h", y=-0.18))
    return style_fig(fig, height=500)


def _curves_fig(curves: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    c = curves.dropna(subset=["m2_cum_pct"])
    fig.add_trace(go.Scatter(x=c["k"], y=c["m2_hi95"], mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=c["k"], y=c["m2_lo95"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(28,122,61,0.15)", name=f"{TM.M2_SHORT} 95% interval",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=c["k"], y=c["m2_cum_pct"], mode="lines", name=TM.M2_SHORT,
                             line=dict(color=C.NPHCDA_GREEN, width=2.4)))
    m1 = curves.dropna(subset=["m1_cum_pct"])
    fig.add_trace(go.Scatter(x=m1["k"], y=m1["m1_cum_pct"], mode="lines", name=TM.M1_SHORT,
                             line=dict(color=C.NAVY, width=2.4, dash="dash")))
    for s in TM.SHARES:
        fig.add_hline(y=s, line=dict(color="#94A3B8", width=1, dash="dot"))
    fig.update_layout(title="Concentration of zero-dose children: cumulative share by number of LGAs",
                      xaxis_title="Number of LGAs (ranked by estimated children)",
                      yaxis_title="Cumulative % of national burden", yaxis=dict(range=[0, 100]),
                      legend=dict(orientation="h", y=-0.18))
    return style_fig(fig, height=460)


def _scatter_fig(x, y, text, xt, yt, title, log=False) -> go.Figure:
    fig = go.Figure()
    lo, hi = float(np.nanmin([np.nanmin(x), np.nanmin(y)])), float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="#9AA8B2"),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", text=text, showlegend=False,
                             marker=dict(size=6, color=C.STEEL, opacity=0.7, line=dict(color="white", width=0.4)),
                             hovertemplate="%{text}<br>%{x:,.1f} vs %{y:,.1f}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt)
    if log:
        fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
    return style_fig(fig, height=460)


def _workbook(lists: dict, notes: list[str]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in lists.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
        pd.DataFrame({"NPHCDA zero-dose LGA priority lists": notes}).to_excel(w, sheet_name="Notes", index=False)
    return buf.getvalue()


def render(data: dict):
    domain_banner("_banner_d5.jpg", "Zero-Dose & Hotspots",
                  "Where are zero-dose children most concentrated? Two Bayesian methods estimate the 2026 "
                  "zero-dose rate and number of children for every LGA, with Pareto concentration and "
                  "Getis-Ord Gi* hotspots.")
    tm = TM.load()
    S = tm["summary"]
    choice = st.radio("Estimation method", [TM.M1, TM.M2, "Compare both methods"], index=0,
                      horizontal=True, key="d5_method",
                      help="Both methods use the same NDHS surveys and the same cohort denominator; they "
                           "differ in how the state picture is brought down to LGAs. Compare them side by "
                           "side in the last option.")
    if choice == "Compare both methods":
        _render_compare(tm, data)
        return
    method = choice
    p = TM._p(method)
    ranked = TM.lga_ranked(tm, method)
    res = TM.state_res(tm, method, data["ndhs_long"])
    s = S[p]
    nat = tm["national_zone"].iloc[0]
    n50, n60, n80 = (TM.n_for_share(ranked, x) for x in TM.SHARES)
    top = ranked.iloc[0]
    st.caption(method_text(method))
    kpi_row([
        {"label": "Zero-dose children, 2026", "value": _m(nat[f"{p}_children"]),
         "sub": clean(f"95% CrI {_m(nat[f'{p}_lo95'])} to {_m(nat[f'{p}_hi95'])}; rate {nat[f'{p}_rate']:.1f}%"),
         "color": C.ACCENT},
        {"label": "LGAs estimated", "value": f"{s['lgas']} of 774",
         "sub": "with a 2026 estimate", "color": C.NAVY},
        {"label": "Pareto concentration", "value": f"50% in {n50} LGAs",
         "sub": f"60% in {n60}; 80% in {n80} LGAs", "color": C.GOLD,
         "help": "The smallest number of top-ranked LGAs that together hold 50, 60 and 80 percent of "
                 "all estimated zero-dose children."},
        {"label": "Highest-burden LGA", "value": clean(top["LGA"]),
         "sub": clean(f"{top['State']}: {top['Zero-dose children (est)']:,} children"), "color": C.STEEL},
        {"label": "Convergence", "value": f"R-hat {s['diagnostics']['max_rhat']:.3f}",
         "sub": f"min ESS {s['diagnostics']['min_ess_bulk']:,.0f}; {s['diagnostics']['divergences']} divergences",
         "color": C.STEEL,
         "help": "R-hat below 1.01 with a large effective sample size (ESS) and no divergent transitions "
                 "means the sampler converged."},
    ])

    tabs = st.tabs(["State estimates", "LGA burden and Pareto", "Hotspot maps", "Ranked LGA table",
                    "Validation", "Diagnostics"])

    # ---------------- State
    with tabs[0]:
        if method == TM.M1:
            section("State zero-dose trajectories and forecast",
                    "NDHS survey history (2008-2024) and the Bayesian forecast to 2028.")
            st.plotly_chart(viz.state_trajectories_fig(res), use_container_width=True)
        else:
            section("State zero-dose estimates, 2026",
                    "State rates are the cohort-weighted sum of each state's LGA posterior rates.")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.forest_fig(res), use_container_width=True)
        with c2:
            st.plotly_chart(viz.burden_bars_fig(res), use_container_width=True)
        if method == TM.M1:
            st.plotly_chart(viz.zone_summary_fig(res), use_container_width=True)
        stab = res[["state", "zone", "zd_obs_2024", "nmdhs_2025_26", "zd_pred_2026_mean", "zd_pred_2026_lo95",
                    "zd_pred_2026_hi95", "zd_count_2026", "zd_count_2026_lo95", "zd_count_2026_hi95",
                    "lga_rate_range", "priority_tier"]].copy()
        stab.columns = ["State", "Zone", "NDHS 2024 (%)", "NmDHS 2025-26 (%)", "2026 rate (%)",
                        "Rate low (95%)", "Rate high (95%)", "Zero-dose children 2026", "Children low (95%)",
                        "Children high (95%)", "LGA rate range (%)", "Priority tier"]
        for c in stab.columns[2:7]:
            stab[c] = pd.to_numeric(stab[c], errors="coerce").round(1)
        for c in ["Zero-dose children 2026", "Children low (95%)", "Children high (95%)"]:
            stab[c] = stab[c].round(0).astype(int)
        stab = stab.sort_values("Zero-dose children 2026", ascending=False)
        st.dataframe(stab, use_container_width=True, height=420)
        _download(stab, "Download state estimates (CSV)", f"D5_state_estimates_2026_{p}.csv")
        ai.ai_block(f"d5_state_{p}", f"Zero-Dose & Hotspots - state estimates ({method})",
                    "State zero-dose rate for 2026 with the 95 percent credible interval, the NDHS 2024 and "
                    "NmDHS 2025-26 survey values, the estimated number of zero-dose children and the "
                    "priority tier.", stab.head(20))
        st.divider()
        ai.chat_panel(f"d5_state_chat_{p}", f"State zero-dose estimates ({method})",
                      "State zero-dose rate and children for 2026 with credible intervals and survey values.",
                      stab.to_dict(orient="records"),
                      suggestions=["Which state has the most zero-dose children?",
                                   "Which states are Tier 1?"])

    # ---------------- LGA Pareto
    with tabs[1]:
        section("Pareto concentration of zero-dose children across LGAs",
                "LGAs ranked by estimated zero-dose children, with the 50, 60 and 80 percent cut-offs.")
        st.plotly_chart(_pareto_fig(ranked, clean(
            f"{method.split(':')[0]}: {ranked['Zero-dose children (est)'].sum():,.0f} zero-dose children across "
            f"{len(ranked)} LGAs")), use_container_width=True)
        ps = tm["pareto_scenarios"]
        ptab = pd.DataFrame({
            "Share of national burden": [f"{int(x)}%" for x in ps["share_of_burden"]],
            "LGAs needed": ps[f"{p}_lgas"].astype(int),
            "Share of all LGAs (%)": ps[f"{p}_pct_of_lgas"].round(1),
            "Share held by that list, 95% CrI (%)": [f"{a:.1f}-{b:.1f}" for a, b in
                                                    zip(ps[f"{p}_list_share_lo95"], ps[f"{p}_list_share_hi95"])],
        })
        st.dataframe(ptab, use_container_width=True)
        st.caption(clean(
            "Priority band: A = the LGAs that together hold the first 50 percent of zero-dose children, "
            "B = 50 to 60 percent, C = 60 to 80 percent, D = the long tail (80 to 100 percent). The 95% "
            "credible interval shows how much of the burden the same list of LGAs holds across the "
            "posterior draws."))
        lists = {f"50pct ({n50} LGAs)": ranked[ranked["Burden rank"] <= n50][LIST_COLS],
                 f"60pct ({n60} LGAs)": ranked[ranked["Burden rank"] <= n60][LIST_COLS],
                 f"80pct ({n80} LGAs)": ranked[ranked["Burden rank"] <= n80][LIST_COLS],
                 f"All ranked ({len(ranked)})": ranked[LIST_COLS]}
        st.markdown("**Download the priority lists** (LGAs ranked by the number of zero-dose children):")
        dc = st.columns(5)
        for i, (nm, df) in enumerate(lists.items()):
            with dc[i]:
                _download(df, clean(nm.replace("pct", "% of burden")) + " (CSV)",
                          f"NPHCDA_LGA_{p}_{nm.split(' ')[0]}.csv", key=f"dl_{p}_{i}")
        with dc[4]:
            notes = [method, "",
                     f"Sheets 1-3: the fewest LGAs that hold 50% ({n50}), 60% ({n60}) and 80% ({n80}) of the "
                     f"estimated {ranked['Zero-dose children (est)'].sum():,.0f} zero-dose children in 2026.",
                     "Sheet 4: all ranked LGAs.", "",
                     "Burden rank: ranked by the NUMBER of zero-dose children.",
                     "Low/high (95%): 95% credible interval.",
                     "Cohort: 12-23-month children = state under-five (2024 projection) / 5, shared to LGAs by "
                     "NPC 2022 population.",
                     "Figures are model estimates."]
            st.download_button("All lists as Excel (.xlsx)", _workbook(lists, notes),
                               f"NPHCDA_LGA_Priority_Lists_{p}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key=f"dlx_{p}")
        st.dataframe(highlight_classes(ranked[LIST_COLS], "Priority band", BAND_CELL),
                     use_container_width=True, height=480)
        ai.ai_block(f"d5_pareto_{p}", f"Zero-Dose & Hotspots - LGA Pareto concentration ({method})",
                    f"LGAs ranked by estimated zero-dose children. 50 percent of the burden is in {n50} LGAs, "
                    f"60 percent in {n60} and 80 percent in {n80} LGAs.",
                    ranked[LIST_COLS].head(25))
        st.divider()
        ai.chat_panel(f"d5_pareto_chat_{p}", f"LGA Pareto ({method})",
                      "Top LGAs ranked by estimated zero-dose children with rate, credible interval, "
                      "cumulative share and priority band.",
                      ranked[LIST_COLS].head(80).to_dict(orient="records"),
                      suggestions=["Which LGA has the most zero-dose children?",
                                   "How many LGAs hold 60 percent of the burden?"])

    # ---------------- Maps
    with tabs[2]:
        _render_maps(tm, method, res)

    # ---------------- Ranked table
    with tabs[3]:
        section("Ranked LGA table", f"{len(ranked)} LGAs, ranked by estimated zero-dose children (2026).")
        f1, f2 = st.columns(2)
        sel = f1.selectbox("Filter by state", ["All"] + sorted(ranked["State"].unique()), key=f"d5_rt_state_{p}")
        q = f2.text_input("Search LGA", key=f"d5_rt_q_{p}")
        view = ranked if sel == "All" else ranked[ranked["State"] == sel]
        if q:
            view = view[view["LGA"].str.contains(q, case=False, na=False)]
        st.dataframe(highlight_classes(view, "Priority band", BAND_CELL), use_container_width=True, height=480)
        _download(ranked, "Download full ranked LGA table (CSV)", f"D5_lga_ranked_{p}.csv")
        st.caption(clean(
            "Probability in top 155: the share of posterior draws in which the LGA ranks among the 155 "
            "highest-burden LGAs (the top 20 percent). Values near 100 percent mark LGAs that are in the "
            "priority group whatever the uncertainty."))
        ai.chat_panel(f"d5_lga_chat_{p}", f"Ranked LGA table ({method})",
                      "All ranked LGAs with state, zone, zero-dose rate and children with 95% credible "
                      "intervals, priority band, archetype and the other method's rank.",
                      ranked.head(150).to_dict(orient="records"),
                      suggestions=["List the top 10 LGAs in Kano.", "Which Zamfara LGAs are in band A?"])

    with tabs[4]:
        _render_validation(tm)

    with tabs[5]:
        section("Model convergence diagnostics",
                "Hyperparameters: R-hat near 1.00 and a large effective sample size indicate convergence.")
        st.dataframe(TM.diagnostics(tm, method), use_container_width=True)
        d = s["diagnostics"]
        st.caption(clean(f"Max R-hat {d['max_rhat']:.3f} across all parameters; minimum bulk ESS "
                         f"{d['min_ess_bulk']:,.0f}, tail ESS {d['min_ess_tail']:,.0f}; "
                         f"{d['divergences']} divergent transitions."))
        if method == TM.M2:
            st.markdown(clean(f"**Posterior predictive check.** {S['m2']['survey_ppc_coverage95'] * 100:.0f}% of "
                              "the NDHS state survey observations fall inside the model's 95% predictive "
                              "interval."))
            st.dataframe(tm["m2_ppc"].round(1), use_container_width=True, height=300)


def _render_maps(tm: dict, method: str, res: pd.DataFrame):
    import spatial
    p = TM._p(method)
    section("Getis-Ord Gi* hotspot maps (LGA)",
            "Local spatial clustering of the 2026 zero-dose rate (k=5 nearest neighbours, row-standardized, "
            "999 permutations).")
    st.caption(clean(
        "A 'Hot Spot' is an LGA whose high zero-dose rate, together with its neighbours, is statistically "
        "unlikely to be chance (p<0.01 deep red, then p<0.05 and p<0.10); 'Cold Spot' marks clusters of "
        "low rates; grey is not significant. Use the scroll bar to move down the page; use the + / - "
        "buttons or drag to zoom the map."))
    gi = TM.gi_lga(method)
    gdf = spatial.load_gdf("lga")
    mf = TM.map_frame(tm, method)
    g = gdf.merge(mf[["state_key", "lga_key", "zd_rate", "zd_children"]], on=["state_key", "lga_key"], how="left")
    if gi is not None:
        g = g.merge(gi[["state_key", "lga_key", "gi_class"]], on=["state_key", "lga_key"], how="left")
    else:
        g["gi_class"] = "Not Significant"
    g["gi_class"] = g["gi_class"].fillna("Not Significant")
    view = st.radio("Map", ["Hotspot clusters (Gi*)", "Zero-dose rate (%)", "Zero-dose children"],
                    horizontal=True, key=f"d5_map_view_{p}")
    if view.startswith("Hotspot"):
        fig = viz.choropleth(g, "gi_class", categorical=True, color_map=C.HOTSPOT_COLORS,
                             title=f"LGA zero-dose hotspot clusters, 2026 ({method.split(':')[0]})",
                             legend_title="Gi* class", height=760)
    elif view.startswith("Zero-dose rate"):
        fig = viz.choropleth(g, "zd_rate", categorical=False, range_color=[0, 90],
                             title=f"Estimated zero-dose rate by LGA, 2026 ({method.split(':')[0]})",
                             legend_title="Rate (%)", height=760)
    else:
        fig = viz.choropleth(g, "zd_children", categorical=False, colorscale="YlOrRd",
                             range_color=[0, float(np.nanpercentile(g["zd_children"], 98))],
                             title=f"Estimated zero-dose children by LGA, 2026 ({method.split(':')[0]})",
                             legend_title="Children", height=760)
    st.plotly_chart(fig, use_container_width=True)
    if gi is not None:
        counts = gi["gi_class"].value_counts().to_dict()
        hot = gi[gi["gi_class"].str.contains("Hot", na=False)].sort_values("gi_z", ascending=False)
        st.caption(clean("LGAs per class: " + "; ".join(f"{k} {v}" for k, v in counts.items())))
        ctx = {"class_counts": counts, "top_hotspot_lgas": hot[["state", "lga", "zd_rate", "gi_class"]]
               .head(30).to_dict(orient="records")}
        ai.ai_block(f"d5_hot_{p}", f"Zero-Dose & Hotspots - Gi* hotspot clusters ({method})",
                    "Counts of LGAs in each Gi* category and the strongest hotspot LGAs.", ctx)

    st.divider()
    years = [y for y in C.FORECAST_YEARS if TM.gi_state(method, y) is not None]
    section("Forecast zero-dose hotspots by state" + (", 2026 to 2028" if len(years) > 1 else ", 2026"),
            "Getis-Ord Gi* on the state zero-dose rate (Queen contiguity, GRID3 state boundaries).")
    if not years:
        st.info("State hotspot results are not available.")
        return
    labels = st.checkbox("Show state labels", value=False, key=f"d5_lbl_{p}")
    hot_by_year = {}
    maps = {}
    for yr in years:
        sg = TM.gi_state(method, yr)
        gg = spatial.load_gdf("state").merge(sg[["state_key", "gi_class", "value"]], on="state_key", how="left")
        gg["gi_class"] = gg["gi_class"].fillna("Not Significant")
        maps[yr] = gg
        hot_by_year[yr] = sorted(sg.loc[sg["gi_class"].str.contains("Hot", na=False), "state"].astype(str))
    yr = st.radio("Forecast year", years, horizontal=True, key=f"d5_hot_year_{p}") if len(years) > 1 else years[0]
    st.plotly_chart(viz.choropleth(maps[yr], "gi_class", categorical=True, color_map=C.HOTSPOT_COLORS,
                                   title=f"State zero-dose hotspots {yr}", legend_title="Gi* class",
                                   height=700, labels=labels, label_size=10), use_container_width=True)
    persistent = sorted(set.intersection(*[set(v) for v in hot_by_year.values()]))
    st.caption(clean("Hot-spot states: " + "; ".join(f"{y}: {', '.join(v) or 'none'}" for y, v in hot_by_year.items())))
    ai.ai_block(f"d5_hot_years_{p}", f"Zero-Dose & Hotspots - state hotspots ({method})",
                "Significant hot-spot states for each forecast year and those that stay hot in every year. "
                "Say whether the cluster is persistent and give one targeting implication.",
                {"hot_states_by_year": hot_by_year, "persistent_hot_states": persistent})


def _render_validation(tm: dict):
    section("External validation of both methods",
            "Agreement with the independent NmDHS 2025-26 survey (state level) and with IHME 2018 LGA "
            "estimates.")
    v = tm["validation"].copy()
    v = v[~v["method"].str.contains("all LGAs")]
    show = pd.DataFrame({
        "Method": v["method"], "Check": v["check"],
        "Spearman rho": v["rho"].round(2),
        "95% CI": [f"{a:.2f} to {b:.2f}" for a, b in zip(v["ci_lo"], v["ci_hi"])],
        "N": v["n"].astype(int),
        "MAE (pp)": v["MAE_pp"].round(1), "Bias (pp)": v["bias_pp"].round(1),
        "States with positive within-state rho": [f"{int(a)} of {int(b)}" if pd.notna(a) else "" for a, b in
                                                   zip(v["states_positive"], v["states_tested"])],
    })
    st.dataframe(show, use_container_width=True)
    st_ = tm["state"]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_scatter_fig(st_["nmdhs_2025_26"], st_["m1_rate"], st_["state"],
                                     "NmDHS 2025-26 zero-dose (100 - Penta1, %)", "Method 1, 2026 (%)",
                                     clean(f"Method 1 vs NmDHS 2025-26: rho {v.iloc[0]['rho']:.2f}")),
                        use_container_width=True)
    with c2:
        st.plotly_chart(_scatter_fig(st_["nmdhs_2025_26"], st_["m2_rate"], st_["state"],
                                     "NmDHS 2025-26 zero-dose (100 - Penta1, %)", "Method 2 (SAE), 2026 (%)",
                                     clean(f"Method 2 (SAE) vs NmDHS 2025-26: rho {v.iloc[1]['rho']:.2f}")),
                        use_container_width=True)
    zv = tm["zone_validation"].rename(columns={
        "area": "Zone", "m1_rate": "Method 1 (%)", "m2_rate": "Method 2 (SAE) (%)",
        "m1_state_model_rate": "Method 1 state model (%)", "nmdhs_2025_26": "NmDHS 2025-26 (%)"}).round(1)
    st.markdown("**Zone level**")
    st.dataframe(zv, use_container_width=True)
    st.caption(clean(
        "NmDHS 2025-26 zero-dose is approximated as 100 minus Penta1 coverage among children 12-23 months. "
        "It is an independent survey not used to fit either model. Both methods rank states similarly to the "
        "survey; within states, Method 2 (SAE) agrees much more closely with the IHME LGA pattern because it "
        "models LGA differences directly. See Triangulation & Cross-Checks for the full LGA comparison."))
    ai.ai_block("d5_validation", "Zero-Dose & Hotspots - external validation of both methods",
                "Spearman agreement, MAE and bias of each method against the NmDHS 2025-26 state survey, and "
                "LGA agreement with IHME 2018 overall and within state. Compare the two methods.",
                show.to_dict(orient="records"))
    import evidence_views as EV
    EV.render_sae_evidence(tm)


def _render_compare(tm: dict, data: dict):
    S = tm["summary"]
    nz = tm["national_zone"].copy()
    nat = nz.iloc[0]
    kpi_row([
        {"label": "Method 1, 2026", "value": _m(nat["m1_children"]),
         "sub": clean(f"95% CrI {_m(nat['m1_lo95'])} to {_m(nat['m1_hi95'])}; 773 LGAs"), "color": C.NAVY},
        {"label": "Method 2 (SAE), 2026", "value": _m(nat["m2_children"]),
         "sub": clean(f"95% CrI {_m(nat['m2_lo95'])} to {_m(nat['m2_hi95'])}; 774 LGAs"), "color": C.NPHCDA_GREEN},
        {"label": "National difference", "value": f"{nat['difference_pct']:+.1f}%",
         "sub": clean(f"{nat['difference']:+,.0f} children (Method 2 vs Method 1)"), "color": C.GOLD},
        {"label": "Top-155 overlap", "value": f"{S['overlap_top155']} of 155",
         "sub": f"rank correlation {S['rank_corr']:.2f}", "color": C.STEEL,
         "help": "How many of the 155 highest-burden LGAs (the top 20 percent) are the same under both "
                 "methods, and the Spearman correlation of the two LGA rankings."},
    ])
    tabs = st.tabs(["National and zone", "Pareto 50/60/80", "LGA agreement", "States", "Archetypes",
                    "Validation", "About the methods"])
    with tabs[0]:
        t = pd.DataFrame({
            "Area": nz["area"], "LGAs": nz["lgas_total"],
            "Cohort 12-23 months": nz["cohort_12_23m"].round(0).astype(int),
            "Method 1 children": nz["m1_children"].round(0).astype(int),
            "Method 1 95% CrI": [f"{a:,.0f}-{b:,.0f}" for a, b in zip(nz["m1_lo95"], nz["m1_hi95"])],
            "Method 1 rate (%)": nz["m1_rate"].round(1),
            "Method 2 (SAE) children": nz["m2_children"].round(0).astype(int),
            "Method 2 (SAE) 95% CrI": [f"{a:,.0f}-{b:,.0f}" for a, b in zip(nz["m2_lo95"], nz["m2_hi95"])],
            "Method 2 (SAE) rate (%)": nz["m2_rate"].round(1),
            "Difference (%)": nz["difference_pct"].round(1)})
        st.dataframe(t, use_container_width=True)
        z = nz.iloc[1:]
        fig = go.Figure()
        fig.add_bar(x=z["area"], y=z["m1_children"] / 1e3, name=TM.M1_SHORT, marker_color=C.NAVY,
                    error_y=dict(type="data", symmetric=False, array=(z["m1_hi95"] - z["m1_children"]) / 1e3,
                                 arrayminus=(z["m1_children"] - z["m1_lo95"]) / 1e3))
        fig.add_bar(x=z["area"], y=z["m2_children"] / 1e3, name=TM.M2_SHORT, marker_color=C.NPHCDA_GREEN,
                    error_y=dict(type="data", symmetric=False, array=(z["m2_hi95"] - z["m2_children"]) / 1e3,
                                 arrayminus=(z["m2_children"] - z["m2_lo95"]) / 1e3))
        fig.update_layout(barmode="group", title="Zero-dose children by zone, 2026 (95% credible interval)",
                          yaxis_title="Children (thousands)", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(style_fig(fig, height=440), use_container_width=True)
        _download(t, "Download national and zone comparison (CSV)", "D5_national_zone_both_methods.csv")
        ai.ai_block("d5_cmp_zone", "Zero-Dose & Hotspots - national and zone comparison of both methods",
                    "National and zone zero-dose children in 2026 under Method 1 and Method 2 (SAE), with "
                    "credible intervals and the difference. Say where the methods agree and differ.",
                    t.to_dict(orient="records"))
    with tabs[1]:
        st.plotly_chart(_curves_fig(tm["pareto_curves"]), use_container_width=True)
        ps = tm["pareto_scenarios"]
        t = pd.DataFrame({
            "Share of national burden": [f"{int(x)}%" for x in ps["share_of_burden"]],
            "Method 1 LGAs": ps["m1_lgas"].astype(int),
            "Method 1 % of LGAs": ps["m1_pct_of_lgas"].round(1),
            "Method 1 list share 95% CrI": [f"{a:.1f}-{b:.1f}" for a, b in zip(ps["m1_list_share_lo95"], ps["m1_list_share_hi95"])],
            "Method 2 (SAE) LGAs": ps["m2_lgas"].astype(int),
            "Method 2 (SAE) % of LGAs": ps["m2_pct_of_lgas"].round(1),
            "Method 2 (SAE) list share 95% CrI": [f"{a:.1f}-{b:.1f}" for a, b in zip(ps["m2_list_share_lo95"], ps["m2_list_share_hi95"])]})
        st.dataframe(t, use_container_width=True)
        tk = tm["topk"]
        t2 = pd.DataFrame({
            "Top LGAs": tk["top_k"].astype(int),
            "Method 1 share (%)": tk["m1_share"].round(1),
            "Method 1 95% CrI": [f"{a:.1f}-{b:.1f}" for a, b in zip(tk["m1_lo95"], tk["m1_hi95"])],
            "Method 1 near-certain members": tk["m1_near_certain"].astype(int),
            "Method 2 (SAE) share (%)": tk["m2_share"].round(1),
            "Method 2 (SAE) 95% CrI": [f"{a:.1f}-{b:.1f}" for a, b in zip(tk["m2_lo95"], tk["m2_hi95"])],
            "Method 2 (SAE) near-certain members": tk["m2_near_certain"].astype(int)})
        st.markdown("**Share of the burden held by the top LGAs**")
        st.dataframe(t2, use_container_width=True)
        st.caption(clean("Near-certain members: LGAs that rank in the top group in at least 90 percent of "
                         "posterior draws. Method 2 (SAE) carries LGA-level uncertainty directly, so fewer LGAs "
                         "are near-certain and its lists are best read with the credible interval."))
        ai.ai_block("d5_cmp_pareto", "Zero-Dose & Hotspots - Pareto concentration under both methods",
                    "Number of LGAs holding 50, 60 and 80 percent of zero-dose children under each method, "
                    "with the uncertainty in the share held by each list.", t.to_dict(orient="records"))
    with tabs[2]:
        ct = TM.comparison_table(tm)
        both = ct.dropna(subset=["Method 1 children"])
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_scatter_fig(both["Method 1 children"].astype(float) + 1,
                                         both["Method 2 (SAE) children"].astype(float) + 1,
                                         both["LGA"] + " (" + both["State"] + ")",
                                         "Method 1 children (log scale)", "Method 2 (SAE) children (log scale)",
                                         clean(f"LGA zero-dose children: rank correlation {S['rank_corr']:.2f}"),
                                         log=True), use_container_width=True)
        with c2:
            st.plotly_chart(_scatter_fig(both["Method 1 rate (%)"], both["Method 2 (SAE) rate (%)"],
                                         both["LGA"] + " (" + both["State"] + ")",
                                         "Method 1 rate (%)", "Method 2 (SAE) rate (%)",
                                         "LGA zero-dose rate, 2026"), use_container_width=True)
        f1, f2 = st.columns(2)
        sel = f1.selectbox("Filter by state", ["All"] + sorted(ct["State"].unique()), key="d5_cmp_state")
        flt = f2.radio("Show", ["All LGAs", "In the top 155 under both", "In the top 155 under one method only"],
                       horizontal=True, key="d5_cmp_flt")
        view = ct if sel == "All" else ct[ct["State"] == sel]
        a, b = view["In top 155, Method 1"] == "Yes", view["In top 155, Method 2 (SAE)"] == "Yes"
        if flt.startswith("In the top 155 under both"):
            view = view[a & b]
        elif flt.startswith("In the top 155 under one"):
            view = view[a ^ b]
        st.dataframe(view, use_container_width=True, height=460)
        _download(ct, "Download LGA comparison, both methods (CSV)", "D5_lga_both_methods_774.csv")
        ai.chat_panel("d5_cmp_chat", "LGA comparison of both methods",
                      "Every LGA with its rank, children and rate under Method 1 and Method 2 (SAE), and "
                      "whether it is in the top 155 under each.",
                      ct.head(200).to_dict(orient="records"),
                      suggestions=["Which LGAs are top 155 under both methods?",
                                   "Where do the methods disagree most?"])
    with tabs[3]:
        s = tm["state"]
        t = pd.DataFrame({
            "State": s["state"], "Zone": s["zone"], "LGAs": s["lgas"],
            "NDHS 2024 (%)": s["ndhs_2024"], "NmDHS 2025-26 (%)": s["nmdhs_2025_26"].round(1),
            "Method 1 rate (%)": s["m1_rate"].round(1), "Method 2 (SAE) rate (%)": s["m2_rate"].round(1),
            "Method 1 children": s["m1_children"].round(0).astype(int),
            "Method 2 (SAE) children": s["m2_children"].round(0).astype(int),
            "Method 1 LGA rate range (%)": s["m1_lga_rate_range"],
            "Method 2 (SAE) LGA rate range (%)": s["m2_lga_rate_range"],
            "Difference (%)": s["difference_pct"].round(1)})
        st.dataframe(t, use_container_width=True, height=520)
        st.caption(clean("LGA rate range: the lowest to highest LGA rate within the state - a measure of how "
                         "much each method separates LGAs inside a state."))
        _download(t, "Download state comparison (CSV)", "D5_state_both_methods.csv")
    with tabs[4]:
        a = tm["archetype"]
        t = pd.DataFrame({
            "Contextual profile": a["archetype"].astype(int).astype(str) + ". " + a["archetype_type"],
            "LGAs": a["lgas"], "Cohort share (%)": a["cohort_share"].round(1),
            "Method 1 children": a["m1_children"].round(0).astype(int),
            "Method 1 share (%)": a["m1_share"].round(1), "Method 1 mean rate (%)": a["m1_mean_rate"].round(1),
            "Method 2 (SAE) children": a["m2_children"].round(0).astype(int),
            "Method 2 (SAE) share (%)": a["m2_share"].round(1),
            "Method 2 (SAE) mean rate (%)": a["m2_mean_rate"].round(1)})
        st.dataframe(t, use_container_width=True)
        st.caption(clean("Contextual profiles (archetypes) come from Ward clustering (k=5) of 15 LGA indicators; "
                         "labels describe only what the indicators measure. Dominant barriers, candidate packages "
                         "and robustness checks are on LGA Priority & Archetypes."))
    with tabs[5]:
        _render_validation(tm)
    with tabs[6]:
        st.markdown(f"**{TM.M1}**")
        st.write(method_text(TM.M1))
        st.markdown(f"**{TM.M2}**")
        st.write(method_text(TM.M2))
        st.markdown(clean(
            "**Common basis.** Both methods use the NDHS 2008-2024 state zero-dose series and the same "
            "12-23-month cohort (state under-five 2024 projection / 5, shared to LGAs by NPC 2022 population). "
            "Method 1 extends the state model with DHIS2 routine data to place children within a state; "
            "Method 2 estimates each LGA directly with covariates and spatial smoothing, so it gives "
            "LGA-specific uncertainty. Both are model estimates for decision support."))
