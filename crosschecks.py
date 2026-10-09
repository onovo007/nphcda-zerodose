"""Triangulation & Cross-Checks view - external convergent-validity checks against independent data."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
import ai
import triangulation as T
import data_io as io
from theme import section, kpi_row, clean, domain_banner, style_fig
from models.d5_zerodose import run_state_model, run_lga_burden


def _download(df, label, fname):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), fname, "text/csv")


def _zone_bar(tab):
    fig = go.Figure()
    fig.add_bar(x=tab["Zone"], y=tab["Our model 2026 (%)"], name="Our model (2026 forecast)",
                marker_color=C.NAVY)
    fig.add_bar(x=tab["Zone"], y=tab["Independent 2024 (%)"],
                name="Independent (Umar et al., 2025; 2024)", marker_color=C.NPHCDA_GREEN)
    fig.update_layout(barmode="group", yaxis_title="Zero-dose prevalence (%)",
                      legend=dict(orientation="h", y=1.12, x=0))
    return style_fig(fig, height=430)


def _scatter(x, y, labels, xt, yt, title=None):
    fig = go.Figure()
    mx = max(max(x), max(y)) * 1.1
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines",
                    line=dict(dash="dash", color="#9AA8B2"), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=y, mode="markers" + ("+text" if labels is not None else ""),
                    text=labels, textposition="top center",
                    marker=dict(size=(11 if labels is not None else 6), color=C.GOLD,
                                line=dict(color=C.NAVY, width=1)), showlegend=False)
    fig.update_layout(xaxis_title=xt, yaxis_title=yt)
    if title:
        fig.update_layout(title=title)
    fig.update_xaxes(range=[0, mx]); fig.update_yaxes(range=[0, mx])
    return style_fig(fig, height=430)


def _matrix(ct):
    dist = [[abs(i - j) for j in range(3)] for i in range(3)]
    fig = go.Figure(go.Heatmap(z=dist, x=[0, 1, 2], y=[0, 1, 2], zmin=0, zmax=2, showscale=False,
                               colorscale=[[0, "#D7EBDD"], [0.5, "#F6E7C7"], [1, "#F2D2CD"]],
                               hoverinfo="skip"))
    rows = list(ct.index)
    for i in range(3):
        for j in range(3):
            fig.add_annotation(x=j, y=i, text=str(int(ct.iloc[i, j])), showarrow=False,
                               font=dict(size=18, color="#1A1A1A"))
    tt = ["Lower (lowest coverage)", "Middle", "Upper (highest coverage)"]
    fig.update_xaxes(tickvals=[0, 1, 2], ticktext=tt, title="IHME DTP1 tercile (2018)")
    fig.update_yaxes(tickvals=[0, 1, 2], ticktext=tt, autorange="reversed",
                     title="Our DTP1 tercile (2026)")
    return style_fig(fig, height=430)


def render(data: dict):
    domain_banner("_banner_d5.jpg", "Triangulation & Cross-Checks",
                  "Do independent datasets agree with our model on WHERE the zero-dose burden "
                  "concentrates? We cross-check the zone forecast against an independent survey "
                  "synthesis, and the LGA estimate against IHME modelled coverage.")

    needed = {"ndhs_long", "under5", "dhis2", "lga_population"}
    if not data or any(data.get(k) is None for k in needed):
        st.warning("Cross-checks need the same inputs as Zero-Dose & Hotspots (NDHS longitudinal, "
                   "under-five, DHIS2, LGA population). Load the bundled sample data or upload them.")
        return

    from models import d5_two_methods as TM
    if TM.is_bundled(data):
        render_two(data)
        return

    kd, kn, ku, kp = (io.df_hash(data["dhis2"]), io.df_hash(data["ndhs_long"]),
                      io.df_hash(data["under5"]), io.df_hash(data["lga_population"]))
    mkey = f"{kn}-{ku}-{kd}-{C.MCMC_DRAWS_LIVE}-{C.MCMC_TUNE_LIVE}"
    with st.spinner("Fitting the state model (cached after the first run)..."):
        res = run_state_model(data["ndhs_long"], data["under5"], data["dhis2"], key=mkey,
                              draws=C.MCMC_DRAWS_LIVE, tune=C.MCMC_TUNE_LIVE)["res"]
    with st.spinner("Building LGA estimates..."):
        clean_df = run_lga_burden(data["dhis2"], res, data["lga_population"], key=f"{mkey}-{kp}")["clean"]

    st.caption(clean("Convergent validity: when independent data, built with different methods, point "
                     "to the same places, confidence in the targeting is high. This is the same logic "
                     "as our Domain 5 vs GBD mortality agreement (r = 0.87)."))

    tabs = st.tabs(["Zone level (independent survey synthesis)",
                    "LGA level (IHME DTP1 coverage)"])

    # ---------- Tab 1: zone ----------
    with tabs[0]:
        zc = T.zone_crosscheck(res)
        section("Zone cross-check: our zone zero-dose vs an independent estimate",
                "Our 2026 zone forecast against the independent zone estimate of Umar et al. (2025).")
        kpi_row([
            {"label": "Rank agreement", "value": f"rho {zc['rho']:.2f}",
             "sub": clean(f"Spearman, {zc['n']} zones, {T.pfmt(zc['p'])}"), "color": C.NPHCDA_GREEN,
             "help": "Spearman rank correlation: 1.00 means the two sources rank the zones identically. "
                     "The p-value is the chance of seeing this agreement if the rankings were unrelated."},
            {"label": "Highest zone (both)", "value": "North-West",
             "sub": "highest zero-dose in both", "color": C.ACCENT},
            {"label": "Method", "value": "Independent",
             "sub": "different surveys, different team", "color": C.STEEL},
        ])
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_zone_bar(zc["table"]), use_container_width=True)
        with c2:
            st.plotly_chart(_scatter(zc["table"]["Independent 2024 (%)"].tolist(),
                                     zc["table"]["Our model 2026 (%)"].tolist(),
                                     zc["table"]["Zone"].tolist(),
                                     "Independent estimate, 2024 (%)", "Our model, 2026 forecast (%)",
                                     title=f"Rank agreement: rho {zc['rho']:.2f}, {T.pfmt(zc['p'])}"),
                            use_container_width=True)
        st.dataframe(zc["table"], use_container_width=True, hide_index=True)
        st.caption(clean("Absolute levels differ between the two methods (different surveys, reference "
                         "years and averaging); the agreement is on where the burden concentrates "
                         "(convergent validity). Source: " + T.UMAR_CITATION))
        _download(zc["table"], "Download zone cross-check (CSV)", "crosscheck_zone.csv")
        ai.ai_block("xc_zone", "Triangulation - zone cross-check vs independent estimate",
                    "Our zone zero-dose (2026) next to an independent 2024 estimate, and the Spearman "
                    "rank correlation. State whether the two sources agree on the ranking of zones and "
                    "what that means for confidence in the targeting.",
                    {"rho": zc["rho"], "p_value": T.pfmt(zc["p"]),
                     "table": zc["table"].to_dict(orient="records")})

    # ---------- Tab 2: LGA ----------
    with tabs[1]:
        section("LGA cross-check: our 2026 estimate vs IHME DTP1 coverage",
                "Our LGA zero-dose (as DTP1 coverage = 100 - zero-dose) against IHME modelled LGA "
                "DTP1 coverage. Compared on rank and tercile, because IHME's latest year is 2018 and "
                "ours is 2026.")
        try:
            ihme = T.load_ihme_dtp1()
            lc = T.lga_crosscheck(clean_df, ihme)
        except Exception as exc:
            st.error(clean(f"LGA cross-check unavailable: {exc}"))
            return
        kpi_row([
            {"label": "Rank agreement", "value": f"rho {lc['rho']:.2f}",
             "sub": clean(f"Spearman, {lc['n']} LGAs, {T.pfmt(lc['p'])}"), "color": C.NPHCDA_GREEN,
             "help": "Rank correlation of our LGA coverage (2026) vs IHME (2018). The p-value is the "
                     "chance of seeing this agreement if the two rankings were unrelated."},
            {"label": "High confidence", "value": f"{lc['n_high']}",
             "sub": clean(f"{lc['n_high']/lc['n']*100:.0f}% same tercile"), "color": C.NPHCDA_GREEN},
            {"label": "Moderate", "value": f"{lc['n_mod']}",
             "sub": clean(f"{lc['n_mod']/lc['n']*100:.0f}% adjacent tercile"), "color": C.GOLD},
            {"label": "Low confidence", "value": f"{lc['n_low']}",
             "sub": clean(f"{lc['n_low']/lc['n']*100:.0f}% opposite - review"), "color": C.ACCENT,
             "help": "LGAs the two sources place in opposite terciles; flagged for data-quality review."},
        ])
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_matrix(lc["crosstab"]), use_container_width=True)
            st.caption(clean("Green diagonal = both sources agree on the tercile; red corners = "
                             "opposite terciles (low confidence)."))
        with c2:
            st.plotly_chart(_scatter(lc["merged"]["IHME DTP1 coverage 2018 (%)"].tolist(),
                                     lc["merged"]["Our DTP1 coverage 2026 (%)"].tolist(), None,
                                     "IHME DTP1 coverage, 2018 (%)", "Our DTP1 coverage, 2026 (%)",
                                     title=f"Rank agreement: rho {lc['rho']:.2f}, {T.pfmt(lc['p'])}"),
                            use_container_width=True)
        st.markdown("##### LGA confidence table")
        only_low = st.checkbox("Show only low-confidence LGAs (flagged for review)", value=False,
                               key="xc_low_only")
        view = lc["merged"]
        if only_low:
            view = view[view["Confidence"] == "Low confidence"]
        st.dataframe(view, use_container_width=True, height=420, hide_index=True)
        _download(lc["merged"], "Download LGA cross-check (CSV)", "crosscheck_lga_ihme.csv")
        st.caption(clean("Compared on rank and tercile: the 8-year gap (IHME 2018 vs our 2026) "
                         "shifts levels but not the broad geography. Source: " + T.IHME_CITATION))
        ai.ai_block("xc_lga", "Triangulation - LGA cross-check vs IHME DTP1 coverage",
                    "Rank correlation and tercile concordance between our LGA estimate (2026) and IHME "
                    "modelled LGA DTP1 coverage (2018). State the level of agreement, how many LGAs are "
                    "high/moderate/low confidence, and that low-confidence LGAs are flagged for review.",
                    {"rho": lc["rho"], "p_value": T.pfmt(lc["p"]), "n": lc["n"],
                     "high": lc["n_high"], "moderate": lc["n_mod"], "low": lc["n_low"]})


# --------------------------------------------------------------------------------------
# Bundled project data: both Domain 5 methods against independent sources
# --------------------------------------------------------------------------------------
def _two_scatter(x, y1, y2, labels, xt, title):
    fig = go.Figure()
    mx = float(max(max(x), max(y1), max(y2))) * 1.08
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="#9AA8B2"),
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=y1, mode="markers", name="Method 1", text=labels,
                    marker=dict(size=9, color=C.NAVY, line=dict(color="white", width=0.8)),
                    hovertemplate="%{text}<br>survey %{x:.1f}% | Method 1 %{y:.1f}%<extra></extra>")
    fig.add_scatter(x=x, y=y2, mode="markers", name="Method 2 (SAE)", text=labels,
                    marker=dict(size=9, color=C.NPHCDA_GREEN, symbol="diamond", line=dict(color="white", width=0.8)),
                    hovertemplate="%{text}<br>survey %{x:.1f}% | Method 2 %{y:.1f}%<extra></extra>")
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title="Model estimate, 2026 (%)",
                      legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(range=[0, mx]); fig.update_yaxes(range=[0, mx])
    return style_fig(fig, height=470)


def render_two(data: dict):
    from scipy.stats import spearmanr
    from models import d5_two_methods as TM
    tm = TM.load()
    v = tm["validation"]
    st.caption(clean("Convergent validity: when independent data, built with different methods, point to the "
                     "same places, confidence in the targeting is high. Both Domain 5 methods are checked: "
                     "Method 1 (Bayesian hierarchical model with DHIS2-calibrated LGA allocation) and Method 2 "
                     "(Bayesian small-area estimation, SAE)."))
    tabs = st.tabs(["State level (NmDHS 2025-26 survey)", "Zone level (NmDHS and Umar et al.)",
                    "LGA level (IHME DTP1 coverage)"])

    # ---------- State: NmDHS 2025-26
    with tabs[0]:
        section("State check: 2026 estimates vs the NmDHS 2025-26 survey",
                "NmDHS 2025-26 is an independent survey not used to fit either model. Zero-dose is "
                "approximated as 100 minus Penta1 coverage among children 12-23 months.")
        r1, r2 = v.iloc[0], v.iloc[1]
        kpi_row([
            {"label": "Method 1 rank agreement", "value": f"rho {r1['rho']:.2f}",
             "sub": clean(f"95% CI {r1['ci_lo']:.2f}-{r1['ci_hi']:.2f}; {T.pfmt(r1['p_value'])}"), "color": C.NAVY},
            {"label": "Method 1 error", "value": f"MAE {r1['MAE_pp']:.1f} pp",
             "sub": clean(f"bias {r1['bias_pp']:+.1f} pp, 37 states"), "color": C.NAVY},
            {"label": "Method 2 (SAE) rank agreement", "value": f"rho {r2['rho']:.2f}",
             "sub": clean(f"95% CI {r2['ci_lo']:.2f}-{r2['ci_hi']:.2f}; {T.pfmt(r2['p_value'])}"),
             "color": C.NPHCDA_GREEN},
            {"label": "Method 2 (SAE) error", "value": f"MAE {r2['MAE_pp']:.1f} pp",
             "sub": clean(f"bias {r2['bias_pp']:+.1f} pp, 37 states"), "color": C.NPHCDA_GREEN},
        ])
        st_ = tm["state"]
        st.plotly_chart(_two_scatter(st_["nmdhs_2025_26"].tolist(), st_["m1_rate"].tolist(),
                                     st_["m2_rate"].tolist(), st_["state"].tolist(),
                                     "NmDHS 2025-26 zero-dose (100 - Penta1, %)",
                                     "State zero-dose: model 2026 vs NmDHS 2025-26"), use_container_width=True)
        t = pd.DataFrame({"State": st_["state"], "Zone": st_["zone"], "NDHS 2024 (%)": st_["ndhs_2024"],
                          "NmDHS 2025-26 (%)": st_["nmdhs_2025_26"].round(1),
                          "Method 1 (%)": st_["m1_rate"].round(1),
                          "Method 2 (SAE) (%)": st_["m2_rate"].round(1)})
        t["Method 1 - survey (pp)"] = (t["Method 1 (%)"] - t["NmDHS 2025-26 (%)"]).round(1)
        t["Method 2 (SAE) - survey (pp)"] = (t["Method 2 (SAE) (%)"] - t["NmDHS 2025-26 (%)"]).round(1)
        t = t.sort_values("NmDHS 2025-26 (%)", ascending=False)
        st.dataframe(t, use_container_width=True, height=420)
        _download(t, "Download state check (CSV)", "crosscheck_state_nmdhs_2025_26.csv")
        st.caption(clean("Spearman rank correlation with a bootstrap 95% confidence interval; MAE = mean "
                         "absolute error in percentage points; bias = mean of model minus survey."))
        ai.ai_block("xc_state_two", "Triangulation - both methods vs NmDHS 2025-26 (state)",
                    "Rank agreement (Spearman rho with 95% CI), MAE and bias of each Domain 5 method against "
                    "the independent NmDHS 2025-26 state survey, and the states with the largest gaps. "
                    "Compare the methods and say what the agreement means for confidence in targeting.",
                    {"Method 1": r1[["rho", "ci_lo", "ci_hi", "MAE_pp", "bias_pp"]].round(3).to_dict(),
                     "Method 2 (SAE)": r2[["rho", "ci_lo", "ci_hi", "MAE_pp", "bias_pp"]].round(3).to_dict(),
                     "largest_gaps": t.reindex(t["Method 1 - survey (pp)"].abs().sort_values(ascending=False)
                                               .index).head(8).to_dict(orient="records")})

    # ---------- Zone
    with tabs[1]:
        section("Zone check: 2026 estimates vs NmDHS 2025-26 and an independent synthesis",
                "Zone zero-dose under both methods against NmDHS 2025-26 and the zone estimate of Umar "
                "et al. (2025, for 2024).")
        zv = tm["zone_validation"]
        z = pd.DataFrame({"Zone": zv["area"], "Method 1 (%)": zv["m1_rate"].round(1),
                          "Method 2 (SAE) (%)": zv["m2_rate"].round(1),
                          "NmDHS 2025-26 (%)": zv["nmdhs_2025_26"].round(1),
                          "Umar et al. 2024 (%)": zv["area"].map(T.UMAR_ZONE_2024)})
        z = z.sort_values("NmDHS 2025-26 (%)", ascending=False).reset_index(drop=True)
        rows = []
        for ref in ["NmDHS 2025-26 (%)", "Umar et al. 2024 (%)"]:
            for m in ["Method 1 (%)", "Method 2 (SAE) (%)"]:
                rho, _ = spearmanr(z[m], z[ref])
                pz = T.exact_spearman_p(z[m].tolist(), z[ref].tolist())
                rows.append({"Model": m.replace(" (%)", ""), "Reference": ref.replace(" (%)", ""),
                             "Spearman rho": round(float(rho), 2), "Exact p": T.pfmt(pz),
                             "MAE (pp)": round(float((z[m] - z[ref]).abs().mean()), 1)})
        zr = pd.DataFrame(rows)
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            for col, color in [("Method 1 (%)", C.NAVY), ("Method 2 (SAE) (%)", C.NPHCDA_GREEN),
                               ("NmDHS 2025-26 (%)", C.GOLD), ("Umar et al. 2024 (%)", "#94A3B8")]:
                fig.add_bar(x=z["Zone"], y=z[col], name=col.replace(" (%)", ""), marker_color=color)
            fig.update_layout(barmode="group", yaxis_title="Zero-dose (%)", legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(style_fig(fig, height=440), use_container_width=True)
        with c2:
            st.dataframe(zr, use_container_width=True)
            st.caption(clean("With six zones, the exact permutation p-value is used. " + T.UMAR_CITATION))
        st.dataframe(z, use_container_width=True)
        _download(z, "Download zone check (CSV)", "crosscheck_zone_both_methods.csv")
        ai.ai_block("xc_zone_two", "Triangulation - zone check, both methods",
                    "Zone zero-dose under both methods against NmDHS 2025-26 and Umar et al. 2024, with rank "
                    "agreement and MAE. State whether the sources agree on the ordering of zones.",
                    {"table": z.to_dict(orient="records"), "agreement": zr.to_dict(orient="records")})

    # ---------- LGA: IHME
    with tabs[2]:
        section("LGA check: 2026 estimates vs IHME DTP1 coverage (2018)",
                "Each method's LGA zero-dose (as DTP1 coverage = 100 - zero-dose) against IHME modelled "
                "LGA DTP1 coverage, compared on rank and tercile because IHME's latest year is 2018.")
        lv = v[v["check"].str.contains("IHME") & ~v["method"].str.contains("all LGAs")]
        summ = pd.DataFrame({
            "Method": lv["method"], "Comparison": lv["check"].str.replace("IHME DTP1 2018, ", "", regex=False),
            "Spearman rho": lv["rho"].round(2),
            "95% CI": [f"{a:.2f} to {b:.2f}" for a, b in zip(lv["ci_lo"], lv["ci_hi"])],
            "LGAs": lv["n"].astype(int),
            "States with positive within-state rho": [f"{int(a)} of {int(b)}" if pd.notna(a) else ""
                                                      for a, b in zip(lv["states_positive"], lv["states_tested"])]})
        st.dataframe(summ, use_container_width=True)
        st.caption(clean("Overall rho reflects the shared north-south gradient. The within-state comparison "
                         "is the stricter test of whether a method places children correctly inside a state: "
                         "Method 2 (SAE) models LGA differences directly and agrees far more closely with IHME "
                         "within states than Method 1, whose within-state pattern follows DHIS2 reporting."))
        meth = st.radio("Method for the tercile comparison", [TM.M1, TM.M2], horizontal=True, key="xc_lga_m")
        lg = tm["lga"]
        pfx = TM._p(meth)
        cdf = pd.DataFrame({"State": lg["state"], "LGA": lg["lga_clean"], "Zone": lg["zone"],
                            "ZD proxy (%)": lg[f"{pfx}_rate"],
                            "ihme_dtp1": 100.0 - pd.to_numeric(lg["ihme_zd_2018"], errors="coerce")}
                           ).dropna(subset=["ZD proxy (%)"])
        try:
            lc = T.lga_crosscheck(cdf, T.load_ihme_dtp1())
        except Exception as exc:
            st.error(clean(f"LGA cross-check unavailable: {exc}"))
            return
        kpi_row([
            {"label": "Rank agreement", "value": f"rho {lc['rho']:.2f}",
             "sub": clean(f"Spearman, {lc['n']} LGAs, {T.pfmt(lc['p'])}"), "color": C.NPHCDA_GREEN},
            {"label": "High confidence", "value": f"{lc['n_high']}",
             "sub": clean(f"{lc['n_high'] / lc['n'] * 100:.0f}% same tercile"), "color": C.NPHCDA_GREEN},
            {"label": "Moderate", "value": f"{lc['n_mod']}",
             "sub": clean(f"{lc['n_mod'] / lc['n'] * 100:.0f}% adjacent tercile"), "color": C.GOLD},
            {"label": "Low confidence", "value": f"{lc['n_low']}",
             "sub": clean(f"{lc['n_low'] / lc['n'] * 100:.0f}% opposite - review"), "color": C.ACCENT},
        ])
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_matrix(lc["crosstab"]), use_container_width=True)
        with c2:
            st.plotly_chart(_scatter(lc["merged"]["IHME DTP1 coverage 2018 (%)"].tolist(),
                                     lc["merged"]["Our DTP1 coverage 2026 (%)"].tolist(), None,
                                     "IHME DTP1 coverage, 2018 (%)", "Model DTP1 coverage, 2026 (%)",
                                     title=f"Rank agreement: rho {lc['rho']:.2f}, {T.pfmt(lc['p'])}"),
                            use_container_width=True)
        only_low = st.checkbox("Show only low-confidence LGAs (flagged for review)", value=False, key="xc_low_two")
        view = lc["merged"].rename(columns={"Our zero-dose 2026 (%)": "Model zero-dose 2026 (%)",
                                            "Our DTP1 coverage 2026 (%)": "Model DTP1 coverage 2026 (%)",
                                            "Our tercile": "Model tercile"})
        if only_low:
            view = view[view["Confidence"] == "Low confidence"]
        st.dataframe(view, use_container_width=True, height=420)
        _download(view, "Download LGA cross-check (CSV)", f"crosscheck_lga_ihme_{pfx}.csv")
        st.caption(clean("Source: " + T.IHME_CITATION))
        ai.ai_block(f"xc_lga_two_{pfx}", f"Triangulation - LGA check vs IHME ({meth})",
                    "Overall and within-state rank agreement of both methods with IHME 2018 LGA DTP1 coverage, "
                    "and the tercile concordance for the selected method. Compare the methods and note that "
                    "low-confidence LGAs are flagged for review.",
                    {"summary": summ.to_dict(orient="records"), "selected_method": meth,
                     "terciles": {"high": lc["n_high"], "moderate": lc["n_mod"], "low": lc["n_low"]}})
