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
    tt = ["Lower (worst)", "Middle", "Upper (best)"]
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
        st.caption(clean("Levels differ for low-burden zones (method and denominator); the agreement "
                         "is on WHERE the burden concentrates. Source: " + T.UMAR_CITATION))
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
        st.caption(clean("Compared on rank/tercile, not levels: the 8-year gap (IHME 2018 vs our 2026) "
                         "shifts levels but not the broad geography. Source: " + T.IHME_CITATION))
        ai.ai_block("xc_lga", "Triangulation - LGA cross-check vs IHME DTP1 coverage",
                    "Rank correlation and tercile concordance between our LGA estimate (2026) and IHME "
                    "modelled LGA DTP1 coverage (2018). State the level of agreement, how many LGAs are "
                    "high/moderate/low confidence, and that low-confidence LGAs are flagged for review.",
                    {"rho": lc["rho"], "p_value": T.pfmt(lc["p"]), "n": lc["n"],
                     "high": lc["n_high"], "moderate": lc["n_mod"], "low": lc["n_low"]})
