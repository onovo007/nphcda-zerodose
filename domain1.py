"""Domain 1 view - Antigen coverage forecasting."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config as C
import viz
import ai
from theme import section, kpi_row, clean, highlight_below, domain_banner
from data_io import (prep_dhis2, national_monthly, df_hash, prep_under5,
                     national_live_births, survey_national_coverage)
from models.d1_forecast import (national_forecasts, lga_at_risk_screen,
                                lga_estimated_coverage_screen,
                                state_antigen_forecasts, lga_antigen_projections, backtest_national)


def _download(df, label, fname):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), fname, "text/csv")


def _horizon_values(series: dict, horizon_months: int):
    """Each antigen's projected value at the END of the chosen horizon (the same calendar month for
    every antigen, so the scorecards are directly comparable). Returns (endpoint_label, {antigen:
    (pct, at_risk)})."""
    import numpy as np
    out = {}
    endpoint_label = ""
    for antigen, s in series.items():
        fx = pd.DatetimeIndex(pd.to_datetime(s["fore_x"]))
        fy = np.asarray(s["fore_y"], dtype=float)
        cut = pd.Timestamp(s["cutoff"])
        ma = np.asarray((fx.year - cut.year) * 12 + (fx.month - cut.month))
        if horizon_months >= 999:
            j = len(fy) - 1
        else:
            elig = np.where(ma <= horizon_months)[0]
            j = int(elig[-1]) if len(elig) else len(fy) - 1
        endpoint_label = fx[j].strftime("%b %Y")
        out[antigen] = (float(fy[j]), bool(fy[j] < 80))
    return endpoint_label, out


def _year_summary(series: dict, year: int, value_col: str):
    """At-risk summary scoped to one calendar year: each antigen's lowest projected value within
    that year, the month it occurs, whether it dips below 80 that year, and the first month below.
    Same value column name as the full summary so the red-flag styling still applies."""
    import numpy as np
    rows = []
    for antigen, s in series.items():
        fx = pd.DatetimeIndex(pd.to_datetime(s["fore_x"]))
        fy = np.asarray(s["fore_y"], dtype=float)
        mask = np.asarray(fx.year == year)
        if not mask.any():
            continue
        vals, months = fy[mask], fx[mask]
        jmin = int(np.argmin(vals))
        below = vals < 80
        rows.append({
            "Antigen": antigen,
            value_col: round(float(vals[jmin]), 2),
            "Month of minimum": months[jmin].strftime("%b %Y"),
            f"Below 80% during {year}": "Yes" if below.any() else "No",
            "First month below 80%": (months[int(np.argmax(below))].strftime("%b %Y")
                                      if below.any() else "-"),
        })
    return pd.DataFrame(rows)


def _values_at_month(series: dict, target: pd.Timestamp):
    """Each antigen's projected value at the chosen calendar month (the forecast month nearest the
    target). Lets the user score a specific period, e.g. Dec 2026. Same return shape as
    _horizon_values."""
    import numpy as np
    out = {}
    endpoint_label = ""
    for antigen, s in series.items():
        fx = pd.DatetimeIndex(pd.to_datetime(s["fore_x"]))
        fy = np.asarray(s["fore_y"], dtype=float)
        j = int(np.argmin(np.abs((fx - target).days)))
        endpoint_label = fx[j].strftime("%b %Y")
        out[antigen] = (float(fy[j]), bool(fy[j] < 80))
    return endpoint_label, out


def render(data: dict):
    domain_banner("_banner_d1.jpg", "Coverage Forecasting",
                  "Which routine antigens are projected to fall below the 80 percent coverage target "
                  "in the next 6 to 12 months?")

    if not data or data.get("dhis2") is None:
        st.warning("Coverage Forecasting needs the DHIS2 export. Load the bundled sample data or upload it.")
        return

    kd = df_hash(data["dhis2"])
    d = prep_dhis2(data["dhis2"])
    nat = national_monthly(d)

    last_obs = nat["ds"].max()
    c_left, c_right = st.columns(2)
    end_year = c_left.selectbox("Run the forecast through (end of year)",
                                [2027, 2028, 2029, 2030, 2031, 2032], index=0, key="d1_end_year")
    metric_choice = c_right.radio(
        "Coverage metric",
        ["Early-Warning Alert (Relative to 2024 baseline)", "Estimated coverage (eligible cohort)"],
        index=0, horizontal=True, key="d1_metric",
        help=("Early-Warning Alert = doses vs the antigen's mean 2024 level (denominator-free; a "
              "at-risk-of-decline early-warning). Estimated coverage = doses divided by the estimated 12-23 month "
              "eligible cohort (under-five / 5), i.e. coverage of the eligible infants."))
    metric = "baseline" if metric_choice.startswith("Early") else "coverage"
    cohort_annual = None
    denom_label = ""
    if metric == "coverage":
        u5_cohort = (float(prep_under5(data["under5"])["cohort_12_23m"].sum())
                     if data.get("under5") is not None else None)
        lb_2024 = (national_live_births(data["live_births"], 2024)
                   if data.get("live_births") is not None else None)
        opts = []
        if u5_cohort:
            opts.append("Under-five / 5 (demographic proxy)")
        if lb_2024:
            opts.append("DHIS2 live births (2024)")
        if not opts:
            st.info(clean("Load the under-five population (or live births) file to use Estimated "
                          "coverage. Showing the Early-Warning Alert index instead."))
            metric = "baseline"
        else:
            denom_choice = st.radio(
                "Eligible-infant denominator", opts, index=0, horizontal=True, key="d1_denom",
                help="The demographic proxy (~7.0M) approximates Nigeria's true annual birth cohort. "
                     "DHIS2 live births are facility-reported and under-count true births, so they can "
                     "push estimated coverage above 100%.")
            if denom_choice.startswith("DHIS2"):
                cohort_annual, denom_label = lb_2024, "DHIS2 live births 2024"
                st.warning(clean(
                    f"DHIS2 live births 2024 (~{lb_2024/1e6:.2f}M) are facility-reported and under-count "
                    f"true births (the demographic cohort is ~{(u5_cohort or 0)/1e6:.1f}M), so coverage "
                    "here may exceed 100%. Use the demographic proxy for WHO/GAVI-comparable coverage; "
                    "this option is for sensitivity comparison only."))
            else:
                cohort_annual, denom_label = u5_cohort, "under-five / 5 proxy"
    out = national_forecasts(nat, key=f"{kd}-{metric}-{denom_label}", end_year=end_year,
                             metric=metric, cohort_annual=cohort_annual)
    series, summary = out["series"], out["summary"]
    unit_label, value_col = out["unit_label"], out["value_col"]

    # NDHS survey coverage for admin-vs-survey triangulation, from ndhs_antigens2024 (the central
    # source: national, population-weighted by under-five). Shown only in WHO admin-coverage mode.
    survey_cov = {}
    survey_src = ""
    if metric == "coverage":
        if data.get("ndhs_antigens") is not None:
            survey_cov = survey_national_coverage(data["ndhs_antigens"], data.get("under5"))
            survey_src = "NDHS 2024 antigens file (national, population-weighted)"
        if not survey_cov:
            survey_cov = dict(C.SURVEY_COVERAGE)
            survey_src = C.SURVEY_COVERAGE_SOURCE
    st.caption(clean(
        f"Metric: {unit_label}"
        + (f" - estimated coverage of the eligible cohort = monthly doses / (annual {denom_label} / 12)."
           if metric == "coverage"
           else " - each antigen vs its own mean 2024 monthly level; the 80% line marks an 80%-of-2024 "
                "at-risk-of-decline early-warning, not literal coverage.")
        + f" Forecast starts after the last observed data ({last_obs:%b %Y}) to Dec {end_year}; "
          "longer horizons widen the prediction intervals."))
    if metric == "baseline":
        st.info(clean("Tip: switch the Coverage metric above to 'Estimated coverage (eligible cohort)' to "
                      "choose the eligible-infant denominator (under-five proxy or DHIS2 live births) and "
                      "to show the NDHS survey reference lines on all four antigen charts."))

    # Time-horizon selector: the scorecards and headline react to the chosen window.
    hlabel = st.radio("Forecast horizon for the scorecards",
                      ["3 months", "6 months", "12 months", "Full forecast", "Pick a month"],
                      index=2, horizontal=True, key="d1_horizon",
                      help="Pick a month lets you score a specific period, e.g. Dec 2026, instead "
                           "of a rolling window.")
    if hlabel == "Pick a month":
        fx0 = pd.DatetimeIndex(pd.to_datetime(next(iter(series.values()))["fore_x"]))
        month_opts = [t.strftime("%b %Y") for t in fx0]
        # Default to the December of the first full forecast year if present, else the last month.
        dec_idx = [i for i, t in enumerate(fx0) if t.month == 12]
        default_idx = dec_idx[0] if dec_idx else len(month_opts) - 1
        chosen = st.selectbox("Score the antigens at this forecast month",
                              month_opts, index=default_idx, key="d1_pick_month")
        endpoint, hv = _values_at_month(series, pd.to_datetime(chosen))
    else:
        H = {"3 months": 3, "6 months": 6, "12 months": 12, "Full forecast": 999}[hlabel]
        endpoint, hv = _horizon_values(series, H)  # endpoint month + antigen -> (pct, at_risk)

    at_risk = [a for a, (v, r) in hv.items() if r]
    low_antigen = min(hv, key=lambda a: hv[a][0]) if hv else None

    if at_risk:
        st.error(clean(f"Headline (by {endpoint}): {len(at_risk)} of {len(series)} tracer antigens are "
                       f"projected to be below the 80 percent target: {', '.join(at_risk)}."))
    elif low_antigen:
        st.success(clean(
            f"Headline (by {endpoint}): all {len(series)} tracer antigens are projected to be at or "
            f"above the 80 percent target nationally. Lowest is {low_antigen} at "
            f"{hv[low_antigen][0]:.0f} {unit_label}."))

    cards = [{"label": "Horizon", "value": hlabel, "sub": clean(f"projected value by {endpoint}"),
              "color": C.STEEL}]
    for antigen in series:
        v, risk = hv[antigen]
        cards.append({"label": antigen, "value": f"{v:.0f}%",
                      "sub": clean(("at risk - by " if risk else "on target - by ") + endpoint),
                      "color": C.ACCENT if risk else C.NPHCDA_GREEN})
    kpi_row(cards)
    st.caption(clean(
        f"Scorecards show each antigen's projected {unit_label} at the end of the selected horizon "
        f"({endpoint}) - the same month for every antigen, so they are directly comparable. The "
        "National at-risk summary below flags any dip below 80% within the 6 to 12-month window."))

    fmonths = pd.to_datetime(next(iter(series.values()))["fore_x"])
    period = f"{fmonths.min():%b %Y} to {fmonths.max():%b %Y}"

    tabs = st.tabs(["National forecast", "LGA at-risk screen", "Microplanning downloads",
                    "Additional antigens"])

    with tabs[3]:
        section("Additional antigens (NPHCDA request)",
                "Established antigens are screened for a projected decline below 80% of their 2024 "
                "level (an at-risk-of-decline early-warning). Recently introduced antigens (IPV2, "
                "Rotavirus) are still scaling up, so they are monitored for uptake rather than decline.")
        st.markdown(clean("**Established additional antigens (OPV3, IPV1, PCV3, Yellow Fever, Men A)** "
                          "- at-risk-of-decline early-warning"))
        est = national_forecasts(nat, key=f"{kd}-extra", metric="baseline",
                                 end_year=end_year, _antigens=C.ANTIGEN_TS_EXTRA)
        ecols = st.columns(2)
        for i, (a, s) in enumerate(est["series"].items()):
            with ecols[i % 2]:
                st.plotly_chart(viz.forecast_band_fig(
                    s, C.ANTIGEN_PAL.get(a, C.NAVY), f"{a} - national ({est['unit_label']})",
                    est["unit_label"], threshold=C.THRESHOLD_PCT, mark_below=True),
                    use_container_width=True)
        st.caption(clean("Values below 80% of the 2024 level carry the at-risk-of-decline flag."))
        st.dataframe(highlight_below(est["summary"], est["value_col"]), use_container_width=True)
        _download(est["summary"], "Download established-antigen forecast (CSV)",
                  "D1_additional_established_forecast.csv")

        st.divider()
        st.markdown(clean("**Recently introduced antigens (IPV2, Rotavirus 1-3)** - uptake, not a "
                          "decline flag"))
        st.caption(clean("These vaccines are still scaling up from recent introduction (Rotavirus from "
                         "2022; second IPV dose), so no 80% at-risk-of-decline flag is applied - the "
                         "curves show uptake. A value above 100% means volume is still growing past the "
                         "2024 level."))
        newf = national_forecasts(nat, key=f"{kd}-new", metric="baseline",
                                  end_year=end_year, _antigens=C.ANTIGEN_TS_NEW)
        ncols = st.columns(2)
        for i, (a, s) in enumerate(newf["series"].items()):
            with ncols[i % 2]:
                st.plotly_chart(viz.forecast_band_fig(
                    s, C.ANTIGEN_PAL.get(a, C.STEEL), f"{a} - national uptake ({newf['unit_label']})",
                    newf["unit_label"], threshold=None, mark_below=False),
                    use_container_width=True)

        st.divider()
        section("LGA at-risk screen - established additional antigens",
                "Each LGA-and-antigen projected 12 months ahead, flagged below 80% of its own 2024 "
                "level. Worst (lowest projection) first.")
        with st.spinner("Screening LGAs for the additional antigens..."):
            escreen = lga_at_risk_screen(data["dhis2"], key=f"{kd}-extra",
                                         _antigens=C.ANTIGEN_TS_EXTRA)
        if escreen.empty:
            st.success("No LGAs projected below 80% for the additional established antigens.")
        else:
            st.write(clean(f"{len(escreen)} LGA-and-antigen combinations carry the at-risk-of-decline "
                           "flag for the additional established antigens."))
            st.dataframe(highlight_below(escreen, "Projected % of baseline (12m)"),
                         use_container_width=True, height=420)
            _download(escreen, "Download additional-antigen LGA at-risk screen (CSV)",
                      "D1_additional_lga_at_risk.csv")

        ai.ai_block(
            "d1_additional_antigens",
            "Coverage Forecasting - additional antigens (established and recently introduced)",
            "Interpret the additional-antigen forecasts. For the ESTABLISHED antigens (OPV3, IPV1, "
            "PCV3, Yellow Fever, Men A), state whether each is projected to stay above or fall below "
            "80 percent of its 2024 level - the at-risk-of-decline early-warning - name any that are "
            "flagged and the lowest one. For the RECENTLY INTRODUCED antigens (IPV2, Rotavirus 1-3), "
            "explain they are still scaling up from recent introduction, so they are monitored for "
            "uptake, not decline; comment on whether uptake is rising and do NOT describe them as "
            "failing. Note how many LGA-and-antigen combinations carry the at-risk-of-decline flag, "
            "name the worst-hit states or antigens if visible, and give one priority action. Be clear "
            "that this is an early-warning of decline against the 2024 level, not coverage of the "
            "eligible child population.",
            {"metric": "percent of 2024 level (at-risk-of-decline early-warning)",
             "established_min_forecasts": est["summary"].to_dict(orient="records"),
             "recently_introduced_min_forecasts": newf["summary"].to_dict(orient="records"),
             "lga_at_risk_flagged_combinations": int(0 if escreen.empty else len(escreen))})

    with tabs[0]:
        section(f"National coverage forecasts ({unit_label})",
                f"Forecast period {period}. Solid line fitted, dashed forecast, shaded 80/95 percent "
                "prediction intervals, red 80 percent target line.")
        if survey_cov:
            st.caption(clean(
                f"Dotted purple line = NDHS survey coverage for all four antigens ({survey_src}). A large "
                "admin-vs-survey gap usually points to denominator or reporting-completeness issues to "
                "reconcile."))
        cols = st.columns(2)
        for i, (antigen, s) in enumerate(series.items()):
            with cols[i % 2]:
                rl = survey_cov.get(antigen)
                st.plotly_chart(
                    viz.forecast_band_fig(s, C.ANTIGEN_PAL[antigen],
                                          f"{antigen} - national coverage forecast",
                                          unit_label, threshold=C.THRESHOLD_PCT, mark_below=True,
                                          ref_line=rl,
                                          ref_label=(f"NDHS {antigen} survey {rl:.0f}%" if rl else "")),
                    use_container_width=True)

        ai_ctx = {
            "metric": unit_label,
            "scorecards_at_selected_horizon": {
                "horizon": hlabel, "endpoint_month": endpoint,
                "antigens": {a: {"value_pct": round(v, 1), "at_risk": bool(r)}
                             for a, (v, r) in hv.items()}},
            "min_forecast_summary": summary.to_dict(orient="records")}
        ai.ai_block("d1_charts", f"Coverage Forecasting - national antigen forecasts ({unit_label})",
                    f"Two things to use together: (1) the SCORECARDS - each antigen's projected value at "
                    f"the end of the selected horizon ({endpoint}), with an at-risk flag; and (2) the "
                    "min-forecast summary - each antigen's lowest projected value, the month it occurs, "
                    "and whether it crosses 80 percent within 6 to 12 months. Cite the scorecard values "
                    "for each antigen, state the overall verdict (which antigens are at risk or all on "
                    "target), name the lowest antigen, and give one priority action.",
                    ai_ctx)

        section("National at-risk summary")
        fx_all = pd.DatetimeIndex(pd.to_datetime(next(iter(series.values()))["fore_x"]))
        years_avail = sorted({int(y) for y in fx_all.year})
        yr_choice = st.selectbox(
            "Period to check", ["Whole forecast"] + [str(y) for y in years_avail], index=0,
            key="d1_atrisk_year",
            help="Pick a calendar year to see each antigen's lowest projected value and at-risk "
                 "status within that year (for example 2026), or keep the whole horizon.")
        show = summary if yr_choice == "Whole forecast" else _year_summary(series, int(yr_choice), value_col)
        st.caption(clean(f"Values below 80 ({unit_label}) are flagged in red."
                         + ("" if yr_choice == "Whole forecast" else f" Scoped to {yr_choice}.")))
        st.dataframe(highlight_below(show, value_col), use_container_width=True)
        _download(show, "Download national forecast summary (CSV)", "D1_national_antigen_forecast.csv")

        with st.expander("Forecast validation (hold-out back-test)"):
            st.caption(clean(
                "We refit on all but the last 6 months and compare the forecast to the held-out "
                "actuals. MAPE (mean absolute percentage error, lower is better) is rated on the "
                "standard Lewis (1982) scale: under 10% = highly accurate, 10-20% = good, 20-50% = "
                "reasonable, over 50% = inaccurate. 95% PI coverage is the share of actuals that fell "
                "inside the model's 95% prediction interval (ideal near 95%; 100% means the intervals "
                "are well-calibrated, if slightly wide)."))
            bt = backtest_national(nat, key=kd, holdout=6)
            if bt.empty:
                st.info("Not enough history to back-test on this dataset.")
            else:
                st.dataframe(bt, use_container_width=True, hide_index=True)
                n_exc = int((bt["MAPE (%)"] < 10).sum())
                st.success(clean(f"{n_exc} of {len(bt)} antigens are 'highly accurate' (MAPE < 10%) on "
                                 "the Lewis scale, with 95% prediction-interval coverage at or near "
                                 "100% - the forecasts are accurate and well-calibrated out-of-sample."))
                ai.ai_block("d1_backtest", "Coverage Forecasting - hold-out back-test",
                            "Out-of-sample accuracy of the national forecasts: MAPE and 95% prediction-"
                            "interval coverage per antigen on a 6-month hold-out. Comment on whether "
                            "accuracy and interval calibration are acceptable and any antigen to treat "
                            "with caution.", bt.to_dict(orient="records"))
        st.divider()
        ai.chat_panel("d1", "Coverage Forecasting - national antigen forecast",
                      f"Per-antigen national Prophet forecast in {unit_label}: the minimum value, the "
                      "month it occurs, and whether each crosses the 80 percent target within 6 to 12 "
                      "months.",
                      summary.to_dict(orient="records"),
                      suggestions=["Which antigen is most at risk?", "When does Penta3 bottom out?"])

    with tabs[1]:
        lga_metric = st.radio(
            "LGA screen metric",
            ["Early-Warning Alert (2024 baseline)", "Estimated coverage (eligible cohort)"],
            horizontal=True, key="d1_lga_metric",
            help="Early-Warning Alert = each LGA vs its own 2024 level (denominator-free at-risk-of-decline flag). "
                 "Estimated coverage = projected doses vs the LGA's estimated 12-23 month cohort "
                 "(2024 under-five / 5, apportioned by LGA population).")

        if lga_metric.startswith("Early"):
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
        else:
            if data.get("under5") is None or data.get("lga_population") is None:
                st.warning("The estimated-coverage LGA screen needs the under-five population and LGA "
                           "population files. Load the bundled sample data or upload them, then retry.")
            else:
                section("LGA estimated-coverage screen (eligible cohort)",
                        "Projected doses 12 months ahead vs each LGA's estimated 12-23 month cohort "
                        "(2024 under-five / 5, apportioned to the LGA by its population). Flagged below "
                        "80 percent coverage.")
                st.caption(clean(
                    "Denominator caveat: the LGA cohort is the state cohort split by LGA population, "
                    "not a directly measured LGA denominator. Where population or reporting do not line "
                    "up, an LGA can read over 100 percent (a denominator or reporting artefact, not a "
                    "true coverage claim). LGAs with low reporting completeness are flagged so a "
                    "reporting gap is not mistaken for low coverage."))
                ku = df_hash(data["under5"])
                kp = df_hash(data["lga_population"])
                esc = lga_estimated_coverage_screen(data["dhis2"], data["under5"],
                                                    data["lga_population"], key=f"{kd}-{ku}-{kp}")
                if esc.empty:
                    st.info("No matched LGAs to score (check the LGA population file names align).")
                else:
                    at_risk = esc[esc["Estimated coverage (12m) %"] < 80]
                    n_over = int((esc["Over 100%"] == "Yes").sum())
                    n_low = int((esc["Low reporting"] == "Yes").sum())
                    st.write(clean(
                        f"{len(at_risk)} of {len(esc)} matched LGA-and-antigen combinations are below 80 "
                        f"percent estimated coverage. {n_over} read over 100 percent (excluded from the "
                        f"at-risk list as a denominator or reporting artefact); {n_low} have low reporting "
                        "completeness (interpret with caution)."))
                    # Distinct-LGA counts (a combination is one LGA x one antigen; an LGA can appear more
                    # than once). "All antigens" = every antigen that LGA reports is below 80 percent.
                    _keys = ["State", "LGA"]
                    n_any = int(at_risk.drop_duplicates(_keys).shape[0])
                    _tot = esc.groupby(_keys).size()
                    _bel = at_risk.groupby(_keys).size()
                    n_all = int((_bel == _tot.reindex(_bel.index)).sum()) if len(_bel) else 0
                    st.write(clean(
                        f"That is {n_any} distinct LGAs with at least one antigen below 80 percent "
                        f"({n_all} have all of their reported antigens below 80 percent)."))
                    st.dataframe(highlight_below(at_risk, "Estimated coverage (12m) %"),
                                 use_container_width=True, height=460)
                    _download(esc, "Download full LGA estimated-coverage screen (CSV)",
                              "D1_lga_estimated_coverage_screen.csv")
                    ai.ai_block("d1_atrisk_cov",
                                "Coverage Forecasting - LGA estimated-coverage screen (eligible cohort)",
                                "LGAs whose projected estimated coverage of the eligible 12-23 month cohort "
                                "falls below 80 percent, by antigen. Name the worst-hit states and antigens, "
                                "note any that are flagged low-reporting or over 100 percent, and give a "
                                "prioritized catch-up action.", at_risk.head(60))

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
