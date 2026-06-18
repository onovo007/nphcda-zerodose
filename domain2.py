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
from models.d2_dropout import (dropout_forecasts, lasso_drivers, lasso_drivers_inference,
                               state_dropout_forecasts, state_year_observed, state_year_with_forecast)


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

    def _fc_at(s, H=12):
        fx = pd.DatetimeIndex(pd.to_datetime(s["fore_x"]))
        fy = np.asarray(s["fore_y"], dtype=float)
        cut = pd.Timestamp(s["cutoff"])
        ma = np.asarray((fx.year - cut.year) * 12 + (fx.month - cut.month))
        m = ma <= H
        return float(fy[m][-1]) if m.any() else float(fy[-1])

    def _direction(latest, val, short=False):
        # Negative dropout = later dose recorded at/above the earlier = NO net dropout (not a loss).
        d = val - latest
        if val <= 0 and latest <= 0:
            return "no net dropout" if short else ("no net dropout (negative - later dose recorded "
                                                   "at or above the earlier)")
        if d > 0.5:
            return "worsening" if short else "worsening (dropout rising)"
        if d < -0.5:
            return "improving" if short else "improving (dropout falling)"
        return "stable"

    # Summary aligned to the on-screen scorecards (12-month forecast), so the AI cites the same numbers.
    fc_summary = {}
    for s in fc.values():
        latest = float(s["obs_y"][-1]); f12 = _fc_at(s, 12); delta = f12 - latest
        fc_summary[s["label"]] = {
            "latest_observed_pct": round(latest, 2),
            "forecast_pct_at_12m": round(f12, 2),
            "change_pp_vs_latest": round(delta, 2),
            "direction": _direction(latest, f12)}

    latest = {col: nat[col].dropna().iloc[-1] for col in C.DROPOUT_TARGETS if col in nat}
    obs_month = pd.Timestamp(nat["ds"].max()).strftime("%b %Y")
    st.markdown("#### Where dropout is now (latest observed)")
    st.caption(clean(f"Most recent reported dropout per antigen pair - the actual value in the latest "
                     f"data month ({obs_month}), not a forecast. This row does not change with the "
                     "horizon selector below."))
    cards = [{"label": "Dropout pairs", "value": str(len(fc)), "sub": "antigen transitions", "color": C.NAVY}]
    for col, label in C.DROPOUT_TARGETS.items():
        if col in latest:
            cards.append({"label": label, "value": f"{latest[col]:.2f}%",
                          "sub": clean(f"latest observed ({obs_month})"),
                          "color": C.DROPOUT_COLORS[col]})
    kpi_row(cards)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    # Time-horizon selector: forecast dropout for the next 3 / 6 / 12 months (micro-planning).
    st.markdown("#### Where dropout is heading (Prophet forecast)")
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
        arrow = _direction(float(s["obs_y"][-1]), val, short=True)
        hcards.append({"label": s["label"], "value": f"{val:.2f}%",
                       "sub": clean(f"forecast at +{H}m ({'+' if delta >= 0 else ''}{delta:.2f} pp vs "
                                    f"latest - {arrow})"),
                       "color": C.DROPOUT_COLORS[col]})
    st.caption(clean(f"Projected dropout rate {hlabel} ahead, per antigen pair, for micro-planning. "
                     "'pp vs latest' is the change in percentage points from the latest observed value "
                     "above (positive = dropout rising, i.e. worsening)."))
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
                    "For each of the three antigen-pair dropout rates you are given the latest observed "
                    "value and the forecast 12 months ahead (these match the scorecards on screen), the "
                    "change in percentage points (change_pp_vs_latest; positive = dropout rising = "
                    "worsening) and the direction. Use ONLY these numbers - cite the forecast_pct_at_12m "
                    "value, not any other horizon. IMPORTANT: a NEGATIVE dropout means the later dose "
                    "was recorded at or above the earlier - i.e. NO net dropout (a reporting/coverage "
                    "signal, not a loss); do NOT describe a pair as 'worsening' if both its values are "
                    "negative - use its stated 'direction' verbatim. Only a positive, rising dropout is "
                    "a genuine concern. State each pair's direction and magnitude, name the pair of "
                    "greatest concern (or say none shows real dropout if all are negative), and give one "
                    "priority action.",
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

            st.divider()
            section("Driver stability and effect size (parsimonious model per pair)",
                    "Bootstrap LASSO selection stability, then a parsimonious model on the top 4 most "
                    "stably selected drivers per pair, reporting the standardized coefficient, its "
                    "direction and a 95 percent confidence interval.")
            st.caption(clean(
                "Uncertainty quantification, not a significance verdict: with 37 states (one row per "
                "state) these are ecological, directional associations to prioritize intervention "
                "hypotheses - not causal effects. Read selection frequency (how often LASSO keeps a "
                "driver) with the coefficient direction and CI width. Dropout can be negative, so a "
                "linear model with robust SEs is used (Beta applies to the bounded zero-dose outcome)."))
            if st.button("Run driver stability + effect-size models", key="d2_infer_btn"):
                st.session_state["d2_infer"] = lasso_drivers_inference(agg, data["model_dataset"],
                                                                       key=f"{kd}-infer")
            inf = st.session_state.get("d2_infer")
            if inf:
                for pair, tbl in inf.items():
                    st.markdown(f"**{clean(pair)}**")
                    st.dataframe(tbl, use_container_width=True, hide_index=True)
                ai.ai_block("d2_drivers_infer", "Dropout & Completion - driver stability and effect sizes",
                            "For each of the three dropout pairs: the top stably-selected drivers, their "
                            "standardized coefficient, direction (higher/lower dropout) and 95% CI. Frame "
                            "as directional associations with uncertainty (not causal), name the most "
                            "stable driver per pair and the intervention it points to, and note when a CI "
                            "is wide or crosses zero. These are ecological state-level associations.",
                            {p: t.to_dict(orient="records") for p, t in inf.items()})

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

    # Per-state dropout (observed) so the chat can answer "which states have the highest dropout".
    state_dropout = {}
    try:
        last_obs_year = int(agg["year"].max())
        for col, label in C.DROPOUT_TARGETS.items():
            if col in agg.columns:
                piv = state_year_observed(agg, col)
                if last_obs_year in piv.columns:
                    state_dropout[label] = {
                        "year": last_obs_year,
                        "top_states_by_dropout": piv[last_obs_year].sort_values(ascending=False)
                        .head(10).round(1).to_dict()}
    except Exception:
        pass

    st.divider()
    ai.chat_panel("d2", "Dropout & Completion - dropout dynamics, drivers and state pattern",
                  "Three things: (1) latest observed and forecast endpoint for each dropout pair; "
                  "(2) the top LASSO drivers per pair; and (3) per-state dropout for the latest observed "
                  "year (top states per pair). State-by-state FORECAST values for 2026/2027 are produced "
                  "on demand in the State-year heatmap tab and are NOT in this context - if asked for a "
                  "forecast year not provided, say it must be generated there and answer from the latest "
                  "observed year instead; never invent state values.",
                  {"forecasts": fc_summary, "drivers": drivers_summary,
                   "state_dropout_latest_observed": state_dropout},
                  suggestions=["Which states have the highest Measles1 to Measles2 dropout?",
                               "What drives Penta1 to Penta3 dropout?"])
