"""Domain 2 view - Dropout and completion dynamics."""
from __future__ import annotations

import streamlit as st

import config as C
import viz
import ai
from theme import section, kpi_row, clean
from data_io import prep_dhis2, national_monthly, state_monthly, df_hash
from models.d2_dropout import (dropout_forecasts, lasso_drivers, state_dropout_forecasts,
                               state_year_observed, state_year_with_forecast)


def _download(df, label, fname):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), fname, "text/csv")


def render(data: dict):
    st.markdown("## Domain 2 - Dropout and completion dynamics")
    st.caption(clean("Research question: what are the predicted dropout rates between key antigen "
                     "pairs, and what factors drive incomplete vaccination? Prophet forecasts plus "
                     "LASSO-selected drivers."))

    if not data or data.get("dhis2") is None:
        st.warning("Domain 2 needs the DHIS2 export. Load the bundled sample data or upload it.")
        return

    kd = df_hash(data["dhis2"])
    d = prep_dhis2(data["dhis2"])
    nat = national_monthly(d)
    agg = state_monthly(d)
    fc = dropout_forecasts(nat, key=kd)

    latest = {col: nat[col].dropna().iloc[-1] for col in C.DROPOUT_TARGETS if col in nat}
    cards = [{"label": "Dropout pairs", "value": str(len(fc)), "sub": "Prophet forecast", "color": C.NAVY}]
    for col, label in C.DROPOUT_TARGETS.items():
        if col in latest:
            cards.append({"label": label, "value": f"{latest[col]:.1f}%", "sub": "latest observed",
                          "color": C.DROPOUT_COLORS[col]})
    kpi_row(cards)

    section("Dropout rate forecasts",
            "Penta1 to Penta3, Penta1 to Measles1, and Measles1 to Measles2, with 80/95 percent intervals.")
    cols = st.columns(3)
    for i, (col, s) in enumerate(fc.items()):
        with cols[i % 3]:
            st.plotly_chart(
                viz.forecast_band_fig(s, C.DROPOUT_COLORS[col],
                                      f"{s['label']} dropout", "Dropout rate (%)"),
                use_container_width=True)
    fc_summary = {s["label"]: {"latest_observed_pct": round(s["obs_y"][-1], 1),
                               "forecast_end_pct": round(s["fore_y"][-1], 1)} for s in fc.values()}
    ai.ai_block("d2_forecast", "Domain 2 - dropout rate forecasts",
                "Prophet forecasts of the three antigen-pair dropout rates; positive values mean "
                "children received the earlier dose but not the later one.", fc_summary)

    drivers_summary: dict = {}
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
        ai.ai_block("d2_drivers", "Domain 2 - dropout drivers (LASSO)",
                    "Cross-validated LASSO coefficients (absolute) linking state equity and "
                    "socioeconomic indicators to each dropout pair; larger means stronger association.",
                    drivers_summary)

    section("Dropout by state and year",
            "Observed annual means. Extend with per-state Prophet forecasts on demand.")
    metric = st.selectbox("Dropout pair", list(C.DROPOUT_TARGETS),
                          format_func=lambda k: C.DROPOUT_TARGETS[k])
    add_fc = st.toggle("Add Prophet forecast columns to 2027 (slower)", value=False)
    last_obs = int(agg["year"].max())
    if add_fc:
        piv = state_year_with_forecast(agg, metric, key=kd)
    else:
        piv = state_year_observed(agg, metric)
    st.plotly_chart(viz.dropout_heatmap_fig(piv, f"{C.DROPOUT_TARGETS[metric]} dropout by state and year",
                    last_obs), use_container_width=True)

    section("State-level dropout forecasts (2026-2027) for microplanning",
            "Per-state Prophet forecast of each dropout pair, monthly for 2026 and 2027, to guide "
            "interventions. Runs about 37 states x 3 pairs; allow a couple of minutes.")
    if st.button("Generate state dropout forecasts", key="d2_state_btn"):
        st.session_state["d2_state_df"] = state_dropout_forecasts(data["dhis2"], key=kd)
    sdf = st.session_state.get("d2_state_df")
    if sdf is not None and not sdf.empty:
        st.success(clean(f"{len(sdf):,} state-pair-month rows."))
        st.dataframe(sdf.head(30), use_container_width=True, height=260)
        st.download_button("Download state dropout forecasts (CSV)",
                           sdf.to_csv(index=False).encode("utf-8"),
                           "D2_state_dropout_forecast_2026_2027.csv", "text/csv")

    st.divider()
    ai.chat_panel("d2", "Domain 2 - dropout dynamics and drivers",
                  "Latest observed and forecast endpoint for each dropout pair (Penta1-Penta3, "
                  "Penta1-Measles1, Measles1-Measles2), and the top LASSO drivers per pair.",
                  {"forecasts": fc_summary, "drivers": drivers_summary},
                  suggestions=["Which dropout pair is worsening?",
                               "What drives Penta1 to Penta3 dropout?"])
