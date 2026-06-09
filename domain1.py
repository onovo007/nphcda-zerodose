"""Domain 1 view - Antigen coverage forecasting."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config as C
import viz
import ai
from theme import section, kpi_row, clean, highlight_below
from data_io import prep_dhis2, national_monthly, df_hash
from models.d1_forecast import (national_forecasts, lga_at_risk_screen,
                                state_antigen_forecasts, lga_antigen_projections)


def _download(df, label, fname):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), fname, "text/csv")


def render(data: dict):
    st.markdown("## Domain 1 - Antigen coverage forecasting")
    st.caption(clean("Research question: which routine antigens are projected to fall below the "
                     "80 percent coverage target in the next 6 to 12 months? National Prophet "
                     "forecasts expressed as a percent of the 2024 baseline."))

    if not data or data.get("dhis2") is None:
        st.warning("Domain 1 needs the DHIS2 export. Load the bundled sample data or upload it.")
        return

    kd = df_hash(data["dhis2"])
    d = prep_dhis2(data["dhis2"])
    nat = national_monthly(d)
    out = national_forecasts(nat, key=kd)
    series, summary = out["series"], out["summary"]

    at_risk = summary[summary["Crosses 80% in 6-12m"] == "Yes"]["Antigen"].tolist()
    lowest = summary.loc[summary["Min forecast (% of 2024 baseline)"].idxmin()]
    kpi_row([
        {"label": "Antigens modelled", "value": str(len(series)), "sub": "national Prophet", "color": C.NAVY},
        {"label": "Lowest antigen", "value": clean(lowest["Antigen"]),
         "sub": f"min {lowest['Min forecast (% of 2024 baseline)']:.0f}% of baseline", "color": C.ACCENT},
        {"label": "At risk in 6-12m", "value": str(len(at_risk)) if at_risk else "0",
         "sub": clean(", ".join(at_risk)) if at_risk else "all on target", "color": C.GOLD},
        {"label": "Target line", "value": "80%", "sub": "of 2024 baseline", "color": C.STEEL},
    ])

    section("National coverage forecasts (% of 2024 baseline)",
            "Solid line fitted, dashed forecast, shaded 80/95 percent prediction intervals, red 80 percent target.")
    cols = st.columns(2)
    for i, (antigen, s) in enumerate(series.items()):
        with cols[i % 2]:
            st.plotly_chart(
                viz.forecast_band_fig(s, C.ANTIGEN_PAL[antigen],
                                      f"{antigen} - national coverage forecast",
                                      "% of 2024 baseline", threshold=C.THRESHOLD_PCT, mark_below=True),
                use_container_width=True)

    section("National at-risk summary")
    st.caption(clean("Values below 80 percent of the 2024 baseline are flagged in red."))
    st.dataframe(highlight_below(summary, "Min forecast (% of 2024 baseline)"), use_container_width=True)
    _download(summary, "Download national forecast summary (CSV)", "D1_national_antigen_forecast.csv")
    ai.ai_block("d1_summary", "Domain 1 - national antigen coverage forecast",
                "Each antigen's national Prophet forecast as a percent of its 2024 baseline, with "
                "the minimum forecast value, when it occurs, and whether it crosses the 80 percent "
                "target within 6 to 12 months.", summary)

    section("Microplanning downloads (2026-2027 projections)",
            "Export the modelled antigen projections for microplanning and decision implementation.")
    monthly = out.get("monthly")
    if monthly is not None and not monthly.empty:
        _download(monthly, "Download national monthly forecast - 4 antigens (CSV)",
                  "D1_national_antigen_forecast_monthly.csv")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**State-level forecasts (Prophet)**")
        st.caption(clean("Per-state Prophet forecast for each antigen, monthly for 2026 and 2027. "
                         "Runs about 37 states x 4 antigens; allow a couple of minutes."))
        if st.button("Generate state-level forecasts", key="d1_state_btn"):
            sdf = state_antigen_forecasts(data["dhis2"], key=kd)
            st.session_state["d1_state_df"] = sdf
        sdf = st.session_state.get("d1_state_df")
        if sdf is not None and not sdf.empty:
            st.success(clean(f"{len(sdf):,} state-antigen-month rows. Below-80% values in red."))
            st.dataframe(highlight_below(sdf.head(30), "pct_of_2024_baseline"),
                         use_container_width=True, height=260)
            _download(sdf, "Download state-level forecasts (CSV)",
                      "D1_state_antigen_forecast_2026_2027.csv")
    with c2:
        st.markdown("**LGA-level projections (trend)**")
        st.caption(clean("Fast trend projection for every LGA and antigen, monthly for 2026 and "
                         "2027. Trend method so all 774 LGAs return in seconds (full per-LGA Prophet "
                         "is impractical live)."))
        if st.button("Generate LGA-level projections", key="d1_lga_btn"):
            ldf = lga_antigen_projections(data["dhis2"], key=kd)
            st.session_state["d1_lga_df"] = ldf
        ldf = st.session_state.get("d1_lga_df")
        if ldf is not None and not ldf.empty:
            st.success(clean(f"{len(ldf):,} LGA-antigen-month rows. Below-80% values in red."))
            st.dataframe(highlight_below(ldf.head(30), "pct_of_2024_baseline"),
                         use_container_width=True, height=260)
            _download(ldf, "Download LGA-level projections (CSV)",
                      "D1_lga_antigen_projections_2026_2027.csv")

    section("LGA at-risk screen (fast trend projection)",
            "Linear-trend projection 12 months ahead per LGA and antigen, flagged below 80 percent of "
            "the LGA's 2024 baseline. The full per-LGA Prophet is the on-demand heavy run.")
    if st.button("Run LGA at-risk screen", type="primary"):
        screen = lga_at_risk_screen(data["dhis2"], key=kd)
        if screen.empty:
            st.success("No LGAs projected below the 80 percent target on the fast screen.")
        else:
            st.write(clean(f"{len(screen)} LGA-and-antigen combinations project below 80 percent "
                           "(all flagged in red)."))
            st.dataframe(highlight_below(screen, "Projected % of baseline (12m)"),
                         use_container_width=True, height=460)
            _download(screen, "Download LGA at-risk screen (CSV)", "D1_lga_at_risk_screen.csv")

    st.divider()
    ai.chat_panel("d1", "Domain 1 - national antigen forecast",
                  "Per-antigen national Prophet forecast as a percent of the 2024 baseline: the "
                  "minimum value, the month it occurs, and whether each crosses the 80 percent "
                  "target within 6 to 12 months.",
                  summary.to_dict(orient="records"),
                  suggestions=["Which antigen is most at risk?", "When does Penta3 bottom out?"])
