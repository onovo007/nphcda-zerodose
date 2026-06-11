"""Domain 1 view - Antigen coverage forecasting."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config as C
import viz
import ai
from theme import section, kpi_row, clean, highlight_below, domain_banner
from data_io import prep_dhis2, national_monthly, df_hash
from models.d1_forecast import (national_forecasts, lga_at_risk_screen,
                                state_antigen_forecasts, lga_antigen_projections)


def _download(df, label, fname):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), fname, "text/csv")


def _horizon_minima(series: dict, horizon_months: int) -> dict:
    """For each antigen, the minimum forecast (% of baseline) within the chosen horizon."""
    import numpy as np
    out = {}
    for antigen, s in series.items():
        fx = pd.DatetimeIndex(pd.to_datetime(s["fore_x"]))
        fy = np.asarray(s["fore_y"], dtype=float)
        cut = pd.Timestamp(s["cutoff"])
        ma = (fx.year - cut.year) * 12 + (fx.month - cut.month)
        mask = np.asarray(ma) <= horizon_months
        if not mask.any():
            mask = np.ones(len(fy), dtype=bool)
        sub = fy[mask]
        j = int(sub.argmin())
        month = fx[mask][j].strftime("%b %Y")
        out[antigen] = (float(sub.min()), month, bool(sub.min() < 80))
    return out


def render(data: dict):
    domain_banner("_banner_d1.jpg", "Coverage Forecasting",
                  "Which routine antigens are projected to fall below the 80 percent coverage target "
                  "in the next 6 to 12 months? National Prophet forecasts as a percent of the 2024 baseline.")

    if not data or data.get("dhis2") is None:
        st.warning("Coverage Forecasting needs the DHIS2 export. Load the bundled sample data or upload it.")
        return

    kd = df_hash(data["dhis2"])
    d = prep_dhis2(data["dhis2"])
    nat = national_monthly(d)
    out = national_forecasts(nat, key=kd)
    series, summary = out["series"], out["summary"]

    # Time-horizon selector: the scorecards and headline react to the chosen window.
    hlabel = st.radio("Forecast horizon for the scorecards",
                      ["3 months", "6 months", "12 months", "Full forecast"],
                      index=2, horizontal=True, key="d1_horizon")
    H = {"3 months": 3, "6 months": 6, "12 months": 12, "Full forecast": 999}[hlabel]
    hm = _horizon_minima(series, H)  # antigen -> (min_pct, month_label, at_risk)

    at_risk = [a for a, v in hm.items() if v[2]]
    low_antigen = min(hm, key=lambda a: hm[a][0]) if hm else None

    if at_risk:
        st.error(clean(f"Headline ({hlabel}): {len(at_risk)} of {len(series)} tracer antigens are "
                       f"projected to fall below the 80 percent target: {', '.join(at_risk)}."))
    elif low_antigen:
        st.success(clean(
            f"Headline ({hlabel}): all {len(series)} tracer antigens are projected to stay at or above "
            f"the 80 percent target nationally. Lowest is {low_antigen} at {hm[low_antigen][0]:.0f} "
            f"percent of the 2024 baseline ({hm[low_antigen][1]})."))

    cards = [{"label": f"Horizon", "value": hlabel, "sub": "of 2024 baseline target 80%", "color": C.STEEL}]
    for antigen in series:
        mn, mon, risk = hm[antigen]
        cards.append({"label": antigen, "value": f"{mn:.0f}%",
                      "sub": clean(("at risk - " if risk else "on target - ") + str(mon)),
                      "color": C.ACCENT if risk else C.NPHCDA_GREEN})
    kpi_row(cards)

    fmonths = pd.to_datetime(next(iter(series.values()))["fore_x"])
    period = f"{fmonths.min():%b %Y} to {fmonths.max():%b %Y}"

    tabs = st.tabs(["National forecast", "LGA at-risk screen", "Microplanning downloads"])

    with tabs[0]:
        section("National coverage forecasts (% of 2024 baseline)",
                f"Forecast period {period}. Solid line fitted, dashed forecast, shaded 80/95 percent "
                "prediction intervals, red 80 percent target line.")
        cols = st.columns(2)
        for i, (antigen, s) in enumerate(series.items()):
            with cols[i % 2]:
                st.plotly_chart(
                    viz.forecast_band_fig(s, C.ANTIGEN_PAL[antigen],
                                          f"{antigen} - national coverage forecast",
                                          "% of 2024 baseline", threshold=C.THRESHOLD_PCT, mark_below=True),
                    use_container_width=True)

        ai.ai_block("d1_charts", "Coverage Forecasting - national antigen coverage forecasts (% of 2024 baseline)",
                    "Four national Prophet forecasts (BCG, Penta1, Penta3, Measles1) as a percent of the "
                    "2024 baseline against the 80 percent target. For each antigen give the minimum "
                    "projected value and the month it occurs, and whether it crosses 80 percent within 6 "
                    "to 12 months. State the overall verdict clearly (which antigens are at risk, or that "
                    "all stay above target), name the lowest antigen, and give one priority action.",
                    summary)

        section("National at-risk summary")
        st.caption(clean("Values below 80 percent of the 2024 baseline are flagged in red."))
        st.dataframe(highlight_below(summary, "Min forecast (% of 2024 baseline)"),
                     use_container_width=True)
        _download(summary, "Download national forecast summary (CSV)", "D1_national_antigen_forecast.csv")
        st.divider()
        ai.chat_panel("d1", "Coverage Forecasting - national antigen forecast",
                      "Per-antigen national Prophet forecast as a percent of the 2024 baseline: the "
                      "minimum value, the month it occurs, and whether each crosses the 80 percent "
                      "target within 6 to 12 months.",
                      summary.to_dict(orient="records"),
                      suggestions=["Which antigen is most at risk?", "When does Penta3 bottom out?"])

    with tabs[1]:
        section("LGA at-risk screen (fast trend projection)",
                "Runs automatically on the loaded data: a linear-trend projection 12 months ahead per "
                "LGA and antigen, flagged below 80 percent of the LGA's 2024 baseline. The 'Projection "
                "month' column shows the period of performance.")
        screen = lga_at_risk_screen(data["dhis2"], key=kd)
        if screen.empty:
            st.success("No LGAs projected below the 80 percent target on the fast screen.")
        else:
            st.write(clean(f"{len(screen)} LGA-and-antigen combinations project below 80 percent "
                           "(all flagged in red)."))
            st.dataframe(highlight_below(screen, "Projected % of baseline (12m)"),
                         use_container_width=True, height=460)
            _download(screen, "Download LGA at-risk screen (CSV)", "D1_lga_at_risk_screen.csv")
            ai.ai_block("d1_atrisk", "Coverage Forecasting - LGA at-risk screen",
                        "LGAs whose linear-trend projection 12 months ahead falls below 80 percent of "
                        "their 2024 baseline, by antigen. Name the worst-hit states/LGAs and antigens "
                        "and give a prioritized catch-up action.", screen.head(60))

    with tabs[2]:
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
                             "About 37 states x 4 antigens; allow a couple of minutes."))
            if st.button("Generate state-level forecasts", key="d1_state_btn"):
                st.session_state["d1_state_df"] = state_antigen_forecasts(data["dhis2"], key=kd)
            sdf = st.session_state.get("d1_state_df")
            if sdf is not None and not sdf.empty:
                st.success(clean(f"{len(sdf):,} rows. Below-80% values in red."))
                st.dataframe(highlight_below(sdf.head(30), "pct_of_2024_baseline"),
                             use_container_width=True, height=260)
                _download(sdf, "Download state-level forecasts (CSV)",
                          "D1_state_antigen_forecast_2026_2027.csv")
        with c2:
            st.markdown("**LGA-level projections (trend)**")
            st.caption(clean("Fast trend projection for every LGA and antigen, monthly for 2026 and "
                             "2027. All 774 LGAs return in seconds (full per-LGA Prophet is impractical live)."))
            if st.button("Generate LGA-level projections", key="d1_lga_btn"):
                st.session_state["d1_lga_df"] = lga_antigen_projections(data["dhis2"], key=kd)
            ldf = st.session_state.get("d1_lga_df")
            if ldf is not None and not ldf.empty:
                st.success(clean(f"{len(ldf):,} rows. Below-80% values in red."))
                st.dataframe(highlight_below(ldf.head(30), "pct_of_2024_baseline"),
                             use_container_width=True, height=260)
                _download(ldf, "Download LGA-level projections (CSV)",
                          "D1_lga_antigen_projections_2026_2027.csv")
