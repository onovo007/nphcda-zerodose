"""Domain 2 view - Dropout and completion dynamics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config as C
import viz
import ai
from theme import section, kpi_row, clean, domain_banner
from data_io import prep_dhis2, national_monthly, state_monthly, df_hash
from models.d2_dropout import (dropout_forecasts, lasso_drivers, state_dropout_forecasts,
                               state_year_observed, state_year_with_forecast)


def _download(df, label, fname):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), fname, "text/csv")


def render(data: dict):
    domain_banner("_banner_d2.jpg", "Dropout & Completion",
                  "What are the predicted dropout rates between key antigen pairs, and what factors "
                  "drive incomplete vaccination? Prophet forecasts plus LASSO-selected drivers.")

    if not data or data.get("dhis2") is None:
        st.warning("Dropout & Completion needs the DHIS2 export. Load the bundled sample data or upload it.")
        return

    kd = df_hash(data["dhis2"])
    d = prep_dhis2(data["dhis2"])
    nat = national_monthly(d)
    agg = state_monthly(d)
    fc = dropout_forecasts(nat, key=kd)

    fc_summary = {s["label"]: {"latest_observed_pct": round(s["obs_y"][-1], 1),
                               "forecast_end_pct": round(s["fore_y"][-1], 1),
                               "direction": ("worsening" if s["fore_y"][-1] > s["obs_y"][-1] + 0.5
                                             else "improving" if s["fore_y"][-1] < s["obs_y"][-1] - 0.5
                                             else "stable")} for s in fc.values()}

    latest = {col: nat[col].dropna().iloc[-1] for col in C.DROPOUT_TARGETS if col in nat}
    st.markdown("**Where dropout is now (latest observed)**")
    st.caption(clean("Most recent reported dropout per antigen pair - a fixed snapshot of the current "
                     "situation. This row does not change with the horizon selector below."))
    cards = [{"label": "Dropout pairs", "value": str(len(fc)), "sub": "Prophet forecast", "color": C.NAVY}]
    for col, label in C.DROPOUT_TARGETS.items():
        if col in latest:
            cards.append({"label": label, "value": f"{latest[col]:.1f}%", "sub": "latest observed",
                          "color": C.DROPOUT_COLORS[col]})
    kpi_row(cards)

    # Time-horizon selector: forecast dropout for the next 3 / 6 / 12 months (micro-planning).
    st.markdown("**Where dropout is heading (Prophet forecast)**")
    hlabel = st.radio("Forecast dropout for the next:", ["3 months", "6 months", "12 months"],
                      index=2, horizontal=True, key="d2_horizon")
    H = {"3 months": 3, "6 months": 6, "12 months": 12}[hlabel]
    hcards = []
    for col, s in fc.items():
        fx = pd.DatetimeIndex(pd.to_datetime(s["fore_x"]))
        fy = np.asarray(s["fore_y"], dtype=float)
        cut = pd.Timestamp(s["cutoff"])
        ma = np.asarray((fx.year - cut.year) * 12 + (fx.month - cut.month))
        mask = ma <= H
        val = float(fy[mask][-1]) if mask.any() else float(fy[-1])
        delta = val - float(s["obs_y"][-1])
        hcards.append({"label": s["label"], "value": f"{val:.1f}%",
                       "sub": clean(f"forecast at +{H}m ({'+' if delta >= 0 else ''}{delta:.1f} vs now)"),
                       "color": C.DROPOUT_COLORS[col]})
    st.caption(clean(f"Projected dropout rate {hlabel} ahead, per antigen pair, for micro-planning."))
    kpi_row(hcards)

    tabs = st.tabs(["Dropout forecasts", "Drivers (LASSO)", "State-year heatmap",
                    "Microplanning downloads"])
    drivers_summary: dict = {}

    with tabs[0]:
        section("Dropout rate forecasts",
                "Penta1 to Penta3, Penta1 to Measles1, and Measles1 to Measles2, with 80/95 percent "
                "prediction intervals. Positive values mean the earlier dose was received but not the later one.")
        cols = st.columns(3)
        for i, (col, s) in enumerate(fc.items()):
            with cols[i % 3]:
                st.plotly_chart(
                    viz.forecast_band_fig(s, C.DROPOUT_COLORS[col],
                                          f"{s['label']} dropout", "Dropout rate (%)"),
                    use_container_width=True)
        ai.ai_block("d2_forecast", "Dropout & Completion - dropout rate forecasts",
                    "Prophet forecasts of the three antigen-pair dropout rates (latest observed vs "
                    "forecast end value, with direction). State the trend direction and magnitude for "
                    "each pair, name the pair of greatest concern, and give one priority action. "
                    "Positive values mean the earlier dose was received but not the later one.",
                    fc_summary)

    with tabs[1]:
        section("Drivers of dropout (LASSO)",
                "Cross-validated LASSO coefficients linking state equity covariates to each dropout pair.")
        if data.get("model_dataset") is None:
            st.info("Upload the zero-dose model dataset (equity covariates) to compute LASSO drivers.")
        else:
            drivers = lasso_drivers(agg, data["model_dataset"])
            cols = st.columns(len(drivers) or 1)
            for i, (target, coefs) in enumerate(drivers.items()):
                with cols[i]:
                    if coefs.empty:
                        st.caption(clean(f"{C.DROPOUT_TARGETS[target]}: no non-zero LASSO coefficients."))
                    else:
                        st.plotly_chart(viz.lasso_bars_fig(coefs, C.DROPOUT_TARGETS[target],
                                        C.DROPOUT_COLORS[target]), use_container_width=True)
            drivers_summary = {C.DROPOUT_TARGETS[t]: {k.replace("pct_", "").replace("_", " "): round(float(v), 3)
                               for k, v in c.head(6).items()} for t, c in drivers.items()}
            ai.ai_block("d2_drivers", "Dropout & Completion - dropout drivers (LASSO)",
                        "Cross-validated LASSO coefficients (absolute) linking state equity and "
                        "socioeconomic indicators to each dropout pair; larger means stronger association. "
                        "Name the leading drivers per pair and what they imply for intervention design.",
                        drivers_summary)

    with tabs[2]:
        section("Dropout by state and year",
                "Observed annual means; gaps where a state reported late are interpolated across years "
                "for display. Extend with per-state Prophet forecasts on demand.")
        metric = st.selectbox("Dropout pair", list(C.DROPOUT_TARGETS),
                              format_func=lambda k: C.DROPOUT_TARGETS[k])
        add_fc = st.toggle("Add Prophet forecast columns to 2027 (slower)", value=False)
        last_obs = int(agg["year"].max())
        piv = (state_year_with_forecast(agg, metric, key=kd) if add_fc
               else state_year_observed(agg, metric))
        # Fill late-reporting gaps so the heatmap has no blank cells.
        piv = piv.astype(float).interpolate(axis=1, limit_direction="both").round(1)
        st.plotly_chart(viz.dropout_heatmap_fig(piv, f"{C.DROPOUT_TARGETS[metric]} dropout by state and year",
                        last_obs), use_container_width=True)
        ly = max(piv.columns)
        hm_ctx = {"metric": C.DROPOUT_TARGETS[metric], "latest_year": int(ly),
                  "highest_dropout_states": piv[ly].sort_values(ascending=False).head(8).round(1).to_dict(),
                  "lowest_dropout_states": piv[ly].sort_values().head(5).round(1).to_dict()}
        ai.ai_block("d2_heatmap", f"Dropout & Completion - {C.DROPOUT_TARGETS[metric]} dropout by state and year",
                    "State-by-year dropout matrix for the selected pair. Identify the states with the "
                    "highest dropout in the latest year, note any clear regional pattern, and give one "
                    "priority action for the worst states. Negative values denote net upward visits.",
                    hm_ctx)

    with tabs[3]:
        section("State-level dropout forecasts (2026-2027) for microplanning",
                "Per-state Prophet forecast of each dropout pair, monthly for 2026 and 2027, to guide "
                "interventions. About 37 states x 3 pairs; allow a couple of minutes.")
        if st.button("Generate state dropout forecasts", key="d2_state_btn"):
            st.session_state["d2_state_df"] = state_dropout_forecasts(data["dhis2"], key=kd)
        sdf = st.session_state.get("d2_state_df")
        if sdf is not None and not sdf.empty:
            st.success(clean(f"{len(sdf):,} state-pair-month rows."))
            st.dataframe(sdf.head(30), use_container_width=True, height=260)
            _download(sdf, "Download state dropout forecasts (CSV)",
                      "D2_state_dropout_forecast_2026_2027.csv")

    st.divider()
    ai.chat_panel("d2", "Dropout & Completion - dropout dynamics and drivers",
                  "Latest observed and forecast endpoint for each dropout pair (Penta1-Penta3, "
                  "Penta1-Measles1, Measles1-Measles2), and the top LASSO drivers per pair.",
                  {"forecasts": fc_summary, "drivers": drivers_summary},
                  suggestions=["Which dropout pair is worsening?",
                               "What drives Penta1 to Penta3 dropout?"])
