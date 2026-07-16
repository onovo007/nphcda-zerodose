"""Domain 5 view - Zero-dose modelling and hotspots."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config as C
import names as N
import viz
import ai
import data_io as io
from theme import (section, kpi_row, clean, domain_banner, highlight_classes,
                   TIER_CELL, SEVERITY_CELL)
from models.d5_zerodose import run_state_model, run_lga_burden, loo_wave_validation as run_loo


def _download(df: pd.DataFrame, label: str, fname: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), fname, "text/csv")


# Clean, executive-ready columns for the LGA priority downloads (drop the within-state tier/severity
# measures and internal flags that confuse a general audience).
PRIORITY_COLS = ["Burden rank", "State", "LGA", "Zone", "Zero-dose children (est)",
                 "Zero-dose rate (%)", "Cumulative % of burden", "Priority band"]


def _lga_workbook_bytes(all_df, top20, top80, n20, n80, n_all):
    """Build the 3-sheet (+ notes) Excel workbook of LGA priority lists, as bytes."""
    from io import BytesIO
    notes = pd.DataFrame({"NPHCDA zero-dose LGA priority lists": [
        "Contents:",
        f"  Sheet 1 - All ranked LGAs: all {n_all} reporting LGAs, ranked by estimated zero-dose children.",
        f"  Sheet 2 - Top 20% ({n20} LGAs): the highest-burden fifth; about 62% of all zero-dose children.",
        f"  Sheet 3 - Top {n80} LGAs: about 80% of all zero-dose children.",
        "",
        "How to read:",
        "  Burden rank - ranked by the NUMBER of zero-dose children (a larger LGA can rank higher even",
        "    at a lower rate).",
        "  Zero-dose rate (%) - modelled estimate, capped at 99% (the worst LGAs sit at that ceiling).",
        "  Cumulative % of burden - running share of the national total down to that LGA.",
        "  Priority band - A drives the first 50% of the burden, B is 50-80%, C is the long tail.",
        "",
        f"Note: {n_all} of Nigeria's 774 LGAs are shown; the rest were excluded in data-quality cleaning.",
        "Figures are model estimates, best used as a relative priority ranking.",
    ]})
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        all_df.to_excel(w, sheet_name="1. All ranked LGAs", index=False)
        top20.to_excel(w, sheet_name=f"2. Top 20pct ({n20})", index=False)
        top80.to_excel(w, sheet_name=f"3. Top {n80} (80pct)", index=False)
        notes.to_excel(w, sheet_name="Notes", index=False)
    return buf.getvalue()


def render(data: dict):
    domain_banner("_banner_d5.jpg", "Zero-Dose & Hotspots",
                  "Where are zero-dose children most concentrated, and what local factors contribute? "
                  "Bayesian hierarchical Beta regression of state rates, distributed to LGA burden by "
                  "population, with Getis-Ord Gi* hotspots.")

    needed = {"ndhs_long", "under5", "dhis2", "lga_population"}
    if not data or any(data.get(k) is None for k in needed):
        st.warning("Zero-Dose & Hotspots needs the NDHS longitudinal, under-five population, DHIS2 and LGA "
                   "population files. Load the bundled sample data or upload them on the Data page.")
        return

    full = st.toggle("Full posterior (3000 draws, slower)", value=False,
                     help="Off uses 1000 draws for a fast live run. On matches the notebook exactly.")
    draws = C.MCMC_DRAWS_FULL if full else C.MCMC_DRAWS_LIVE
    tune = C.MCMC_TUNE_FULL if full else C.MCMC_TUNE_LIVE

    kd, kn, ku, kp = (io.df_hash(data["dhis2"]), io.df_hash(data["ndhs_long"]),
                      io.df_hash(data["under5"]), io.df_hash(data["lga_population"]))
    mkey = f"{kn}-{ku}-{kd}-{draws}-{tune}"
    with st.spinner("Fitting Bayesian hierarchical Beta model on the survey data..."):
        out = run_state_model(data["ndhs_long"], data["under5"], data["dhis2"],
                              key=mkey, draws=draws, tune=tune)
    res = out["res"]

    lkey = f"{mkey}-{kp}"
    with st.spinner("Building population-weighted LGA burden..."):
        lga = run_lga_burden(data["dhis2"], res, data["lga_population"], key=lkey)
    clean_df, pareto = lga["clean"], lga["pareto"]

    nat_2026 = res["zd_count_2026"].sum()
    top_state = res.iloc[0]
    n20 = int(round(0.20 * lga["n_lgas"]))
    kpi_row([
        {"label": "National zero-dose, 2026", "value": f"{nat_2026/1e6:.2f}M",
         "sub": "state-model sum", "color": C.ACCENT},
        {"label": "LGA-sum burden", "value": f"{lga['national_total']/1e6:.2f}M",
         "sub": clean(f"{lga['national_total']:,} across {lga['n_lgas']} LGAs"), "color": C.NAVY},
        {"label": "Highest-risk state", "value": clean(top_state["state"]),
         "sub": f"{top_state['zd_pred_2026_mean']:.0f}% predicted 2026", "color": C.GOLD},
        {"label": "Pareto concentration", "value": f"top 20% = {lga['top20_pct']:.0f}%",
         "sub": f"= {n20} LGAs; 80% of burden in {lga['n80']} LGAs", "color": C.STEEL,
         "help": "The top 20% of LGAs by burden (= the listed number of LGAs) hold this share of all "
                 "zero-dose children. Higher means more concentrated, so targeting fewer LGAs reaches "
                 "more children."},
        {"label": "Convergence", "value": f"R-hat {out['max_rhat']}",
         "sub": f"min ESS {out['min_ess']} | {out['n_draws']} draws", "color": C.STEEL,
         "help": "MCMC convergence check. R-hat near 1.00 with high effective sample size (ESS) means "
                 "the Bayesian sampler converged; R-hat above 1.01 would flag non-convergence."},
    ])

    with st.expander("Population sensitivity: 2024 vs 2025 under-five (affects burden counts only)"):
        try:
            pop25 = io.prep_under5(pd.read_csv(C.UNDER5_2025_PATH))
            m25 = dict(zip(pop25["jk"], pop25["cohort_12_23m"]))
            t = res[["state", "zd_pred_2026_mean", "cohort_12_23m", "zd_count_2026"]].copy()
            t["jk"] = (t["state"].astype(str).str.upper().str.replace(" ", "", regex=False)
                       .str.replace(",ABUJA", "", regex=False).str.replace(",", "", regex=False))
            t["cohort_2025"] = t["jk"].map(m25).fillna(t["cohort_12_23m"])
            t["burden_2025"] = t["zd_pred_2026_mean"] / 100 * t["cohort_2025"]
            b24, b25 = float(t["zd_count_2026"].sum()), float(t["burden_2025"].sum())
            cc = st.columns(3)
            cc[0].metric("Burden on 2024 population", f"{b24:,.0f}")
            cc[1].metric("Burden on 2025 population", f"{b25:,.0f}")
            cc[2].metric("Change", f"{(b25 / b24 - 1) * 100:+.1f}%")
            st.caption(clean(
                "Zero-dose rates, priority tiers and rate-based rankings are unchanged - only the cohort "
                "denominator (so the burden counts) differ. 2024 is the published basis; 2025 is shown "
                "for sensitivity. The large single-year state swings suggest the 2025 file is a "
                "re-projection, so verify its source before adopting it as the basis."))
            t["change_pct"] = (t["cohort_2025"] / t["cohort_12_23m"] - 1) * 100
            mv = t.sort_values("change_pct")
            show = pd.concat([mv.head(3), mv.tail(3)])[["state", "cohort_12_23m", "cohort_2025",
                                                        "change_pct"]].copy()
            show.columns = ["State", "Cohort 2024", "Cohort 2025", "Change %"]
            show["Cohort 2024"] = show["Cohort 2024"].round(0).astype(int)
            show["Cohort 2025"] = show["Cohort 2025"].round(0).astype(int)
            show["Change %"] = show["Change %"].round(1)
            st.dataframe(show, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.info(clean(f"Population sensitivity unavailable: {exc}"))

    tabs = st.tabs(["State forecasts", "LGA burden and Pareto", "Hotspot maps",
                    "Ranked LGA table", "Diagnostics"])

    with tabs[0]:
        section("State zero-dose trajectories and forecast",
                "Survey-anchored history (2008-2024) and Bayesian forecast to 2028.")
        st.plotly_chart(viz.state_trajectories_fig(res), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.forest_fig(res), use_container_width=True)
        with c2:
            st.plotly_chart(viz.burden_bars_fig(res), use_container_width=True)
        st.plotly_chart(viz.zone_summary_fig(res), use_container_width=True)
        ai.ai_block("d5_state", "Zero-Dose & Hotspots - state zero-dose forecasts and risk ranking",
                    "Predicted state zero-dose rate for 2026 with 95 percent credible interval, the "
                    "observed 2024 rate, the estimated 2026 burden of children, and the priority tier.",
                    res[["state", "zone", "zd_obs_2024", "zd_pred_2026_mean", "zd_pred_2026_lo95",
                         "zd_pred_2026_hi95", "zd_count_2026", "priority_tier"]].head(20))
        st.divider()
        ai.chat_panel("d5_state_chat", "State zero-dose forecasts",
                      "Predicted state zero-dose rate for 2026 with 95 percent credible interval, "
                      "observed 2024 rate, estimated 2026 burden, and priority tier, for all states.",
                      res[["state", "zone", "zd_obs_2024", "zd_pred_2026_mean", "zd_pred_2026_lo95",
                           "zd_pred_2026_hi95", "zd_count_2026", "priority_tier"]].to_dict(orient="records"),
                      suggestions=["Which state has the highest 2026 burden?", "Which states are Tier 1?"])

    with tabs[1]:
        section("Pareto concentration of zero-dose burden across LGAs",
                "LGAs ranked by estimated burden against the cumulative share of the national total.")
        st.plotly_chart(viz.pareto_fig(pareto, lga["top20_pct"], lga["n80"],
                                       lga["national_total"]), use_container_width=True)
        st.caption(clean(
            "Severity classes (within state): Critical = Tier 1, the highest-burden LGAs (red) - act "
            "first; High = Tier 2 (orange); Moderate = Tier 3 (amber); Lower = Tier 4 (blue). Priority "
            "band: A = drives the first 50 percent of national burden, B = 50-80 percent, C = the long "
            "tail. Use Severity to triage where to deploy catch-up first, and Priority band to size "
            "how many LGAs to cover for a target share of the burden."))
        # Enrich with the LGA archetype and equity-deprivation tier (from the LGA archetype master).
        try:
            import re as _re
            def _nk(s, l):
                return (_re.sub(r"[^a-z0-9]", "", str(s).lower()) + "|"
                        + _re.sub(r"[^a-z0-9]", "", _re.sub(r"local government area|lga", "", str(l).lower())))
            _lut = pd.read_csv(C.DATA_DIR / "lga_priority_ranking.csv")
            _lut["_k"] = [_nk(s, l) for s, l in zip(_lut["State"], _lut["LGA"])]
            _am = dict(zip(_lut["_k"], _lut["Archetype"]))
            _tm = dict(zip(_lut["_k"], _lut["Equity tier"]))
            _pk = [_nk(s, l) for s, l in zip(pareto["State"], pareto["LGA"])]
            pareto["Archetype"] = [_am.get(k) for k in _pk]
            pareto["Equity tier"] = [_tm.get(k) for k in _pk]
        except Exception:
            pass
        col_sev = "Severity (within state)" if "Severity (within state)" in pareto.columns else None
        show = highlight_classes(pareto.head(60), col_sev, SEVERITY_CELL) if col_sev else pareto.head(60)
        st.dataframe(show, use_container_width=True, height=420)
        rank_col = "Burden rank" if "Burden rank" in pareto.columns else pareto.columns[0]
        # Executive-ready download set: clean columns only (drop within-state tier/severity/flags).
        dl_cols = [c for c in PRIORITY_COLS if c in pareto.columns] or list(pareto.columns)
        dl_cols += [c for c in ["Archetype", "Equity tier"] if c in pareto.columns]
        par_all = pareto[dl_cols].copy()
        par_top20 = par_all[pareto[rank_col] <= n20]
        par_top80 = par_all[pareto[rank_col] <= lga["n80"]]
        st.markdown("**Download the priority list** (LGAs ranked by the number of zero-dose children):")
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            _download(par_all, f"All {lga['n_lgas']} LGAs (CSV)", "NPHCDA_LGA_ranked_all.csv")
        with dc2:
            _download(par_top20, f"Top 20% - {n20} LGAs ({lga['top20_pct']:.0f}% of burden) (CSV)",
                      "NPHCDA_LGA_top20pct.csv")
        with dc3:
            _download(par_top80, f"Top {lga['n80']} LGAs (80% of burden) (CSV)",
                      "NPHCDA_LGA_top80pct.csv")
        with dc4:
            xlsx = _lga_workbook_bytes(par_all, par_top20, par_top80, n20, lga["n80"], lga["n_lgas"])
            st.download_button("All three as Excel (.xlsx)", xlsx,
                               "NPHCDA_LGA_Priority_Lists.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        ai.ai_block("d5_pareto", "Zero-Dose & Hotspots - LGA Pareto concentration of zero-dose burden",
                    f"Top LGAs ranked by estimated zero-dose children. Nationally about "
                    f"{lga['national_total']:,} children across {lga['n_lgas']} LGAs; the top 20 "
                    f"percent of LGAs carry about {lga['top20_pct']:.0f} percent of the burden.",
                    pareto.head(25))
        st.divider()
        ai.chat_panel("d5_pareto_chat", "LGA Pareto and burden concentration",
                      "Top LGAs ranked by estimated zero-dose children, with state, zone, population, "
                      "rate, cumulative share of the national burden and priority band.",
                      pareto.head(60).to_dict(orient="records"),
                      suggestions=["Which LGA has the worst burden?",
                                   "How many LGAs hold 80 percent of the burden?"])

    with tabs[2]:
        section("Getis-Ord Gi* hotspot maps",
                "LGA-level local spatial autocorrelation (k=5 nearest neighbours) and estimated rate. "
                "Computed automatically on the loaded data and cached.")
        st.caption(clean(
            "How to read the Gi* map: a 'Hot Spot' is an LGA whose high zero-dose burden, together with "
            "its neighbours, is statistically unlikely to be chance - p<0.01 (deep red) is the most "
            "confident, then p<0.05 and p<0.10. 'Cold Spot' (blues) marks clusters of low burden; 'Not "
            "Significant' (grey) shows no clustering. Target the p<0.01 hot-spot clusters first for "
            "coordinated, multi-LGA response. The right map shows the estimated zero-dose rate per LGA "
            "(green low to red high)."))
        with st.spinner("Computing Getis-Ord Gi* hotspots..."):
            import spatial
            gi = spatial.lga_gi_star(clean_df.rename(columns={
                "State": "state", "LGA": "lga", "ZD proxy (%)": "zd_proxy_pct"})
                [["state", "lga", "zd_proxy_pct"]], key=lkey)
            gdf = spatial.load_gdf("lga").merge(
                gi[["state_key", "lga_key", "zd_proxy_pct", "gi_class"]],
                on=["state_key", "lga_key"], how="left")
            gdf["gi_class"] = gdf["gi_class"].fillna("Not Significant")
        if True:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(viz.choropleth(gdf, "gi_class", categorical=True,
                                color_map=C.HOTSPOT_COLORS, title="LGA zero-dose hotspot clusters (Gi*)",
                                legend_title="Gi* class"), use_container_width=True)
            with c2:
                st.plotly_chart(viz.choropleth(gdf, "zd_proxy_pct", categorical=False,
                                range_color=[0, 99], title="Estimated zero-dose rate by LGA (%)",
                                legend_title="ZD rate (%)"), use_container_width=True)
            hot = gi[gi["gi_class"].str.contains("Hot", na=False)].sort_values("gi_z", ascending=False)
            hot_ctx = {"class_counts": gi["gi_class"].value_counts().to_dict(),
                       "top_hotspot_lgas": hot[["state", "lga", "zd_proxy_pct", "gi_class"]]
                       .head(30).to_dict(orient="records")}
            ai.ai_block("d5_hot", "Zero-Dose & Hotspots - Getis-Ord Gi* hotspot clusters",
                        "Counts of LGAs in each Gi* category (statistically significant hot spots and "
                        "cold spots of zero-dose burden, k=5 nearest neighbours), and the strongest "
                        "hotspot LGAs.", hot_ctx)
            st.divider()
            ai.chat_panel("d5_hot_chat", "Gi* hotspot clusters",
                          "Counts of LGAs per Gi* category and the strongest hotspot LGAs by z-score.",
                          hot_ctx,
                          suggestions=["Which LGAs are the strongest hotspots?",
                                       "How many hot spots at p<0.01?"])

        st.divider()
        section("Forecast zero-dose hotspots, 2026 to 2028 (state-level)",
                "Figure 11. Getis-Ord Gi* on the Bayesian posterior predicted state zero-dose means "
                "(Queen contiguity, GRID3 admin-1 boundaries). Shows whether the hot cluster persists "
                "across the forecast horizon.")
        st.caption(clean(
            "Each map is one forecast year. A 'Hot Spot' is a state whose high predicted zero-dose "
            "rate, together with its neighbours, is statistically unlikely to be chance (p<0.01 deep "
            "red is most confident, then p<0.05, p<0.10); 'Cold Spot' (blues) marks low-burden "
            "clusters; grey is not significant. A cluster that stays put year on year is a structural, "
            "multi-year problem, not a one-year spike."))
        res_keyed = res.copy()
        res_keyed["state_key"] = res_keyed["state"].map(N.nstate)
        year_maps, hot_by_year = {}, {}
        for yr in C.FORECAST_YEARS:
            vcol = f"zd_pred_{yr}_mean"
            if vcol not in res_keyed.columns:
                continue
            sv = res_keyed[["state_key", vcol]].rename(columns={vcol: "value"})
            sg = spatial.state_gi_star(sv, key=f"{mkey}-stategi-{yr}", value_col="value")
            g = spatial.load_gdf("state").merge(sg[["state_key", "gi_class", "value"]],
                                                on="state_key", how="left")
            g["gi_class"] = g["gi_class"].fillna("Not Significant")
            year_maps[yr] = g
            hot_by_year[yr] = sorted(sg.loc[sg["gi_class"].str.contains("Hot", na=False), "state"]
                                     .astype(str).tolist())
        if year_maps:
            yrs = list(year_maps.keys())
            vc1, vc2 = st.columns([2, 1])
            view = vc1.radio("Map view", ["Large (one year at a time)", "Compare all three years"],
                             horizontal=True, key="d5_hot_year_view",
                             help="Large shows one full-width map you can read clearly; Compare shows "
                                  "all three years side by side to see how the cluster moves.")
            show_labels = vc2.checkbox("Show state labels", value=False, key="d5_hot_labels",
                                       help="Overlay each state's name on the map.")
            if view.startswith("Compare"):
                ycols = st.columns(len(year_maps))
                for i, (yr, g) in enumerate(year_maps.items()):
                    with ycols[i]:
                        st.plotly_chart(
                            viz.choropleth(g, "gi_class", categorical=True, color_map=C.HOTSPOT_COLORS,
                                           title=f"{yr}", legend_title="Gi* class", height=460,
                                           labels=show_labels, label_size=7),
                            use_container_width=True)
            else:
                yr_sel = st.radio("Forecast year", yrs, horizontal=True, key="d5_hot_year_sel")
                st.plotly_chart(
                    viz.choropleth(year_maps[yr_sel], "gi_class", categorical=True,
                                   color_map=C.HOTSPOT_COLORS,
                                   title=f"Zero-dose hotspots {yr_sel}", legend_title="Gi* class",
                                   height=720, labels=show_labels, label_size=10),
                    use_container_width=True)
            persistent = sorted(set.intersection(*[set(v) for v in hot_by_year.values()])) \
                if hot_by_year else []
            yctx = {"hot_states_by_year": hot_by_year, "persistent_hot_states": persistent}
            ai.ai_block("d5_hot_years",
                        "Zero-Dose & Hotspots - forecast state hotspots 2026 to 2028",
                        "Statistically significant zero-dose hot-spot states for each forecast year and "
                        "the states that stay hot across all years. State whether the cluster is "
                        "persistent (structural) or shifting, name the persistent hot states, and give "
                        "one targeting implication.", yctx)

    with tabs[3]:
        section("Ranked LGA table (population-weighted)",
                f"{lga['n_lgas']} reporting LGAs. Population matched for {lga['matched_pop']} LGAs.")
        f1, f2 = st.columns([1, 1])
        states = ["All"] + sorted(clean_df["State"].unique())
        sel = f1.selectbox("Filter by state", states)
        q = f2.text_input("Search LGA")
        view = clean_df if sel == "All" else clean_df[clean_df["State"] == sel]
        if q:
            view = view[view["LGA"].str.contains(q, case=False, na=False)]
        st.caption(clean("LGA tier colours: Tier 1 Critical (red), Tier 2 High (orange), Tier 3 "
                         "Moderate (amber), Tier 4 Lower (blue). Act on Tier 1 LGAs first. The 'ZD "
                         "count low/high (95%)' columns carry the state posterior credible interval "
                         "down to each LGA (allocation of the state uncertainty, not an independent "
                         "LGA model)."))
        show = highlight_classes(view, "LGA tier", TIER_CELL) if "LGA tier" in view.columns else view
        st.dataframe(show, use_container_width=True, height=480)
        _download(clean_df, "Download full ranked LGA table (CSV)", "D5_lga_zero_dose_ranked_CLEAN.csv")
        ai.ai_block("d5_lga", "Zero-Dose & Hotspots - ranked LGA table (population-weighted)",
                    "The highest-burden LGAs with state, zone, estimated zero-dose rate and count, "
                    "tier and within-state severity.", clean_df.head(25))
        st.divider()
        ai.chat_panel("d5_lga_chat", "Ranked LGA table (population-weighted)",
                      "All ranked LGAs with state, zone, estimated zero-dose rate and count, tier "
                      "and within-state severity. This is the LGA-level source of truth.",
                      clean_df.head(120).to_dict(orient="records"),
                      suggestions=["Which LGA has the worst burden?", "List the top 10 LGAs in Kano."])

    with tabs[4]:
        section("Model convergence diagnostics", "R-hat near 1.00 and adequate ESS indicate convergence.")
        st.dataframe(out["diag"], use_container_width=True)
        st.caption(clean(f"Max R-hat {out['max_rhat']} | min bulk ESS {out['min_ess']} | "
                         f"{out['n_draws']} posterior draws ({'full' if full else 'live'} sampler)."))

        st.divider()
        section("Out-of-sample validation (leave-one-wave-out)",
                "Refit the model with each NDHS survey wave held out, predict that wave, and compare "
                "to observed: MAE/RMSE in percentage points and 95% credible-interval coverage.")
        st.caption(clean("This refits the Bayesian model once per survey wave, so it takes a few "
                         "minutes. It tests whether the time trend generalizes to an unseen survey round."))
        if st.button("Run leave-one-wave-out validation", key="d5_loo_btn"):
            with st.spinner("Refitting the model for each held-out wave..."):
                st.session_state["d5_loo"] = run_loo(data["ndhs_long"], data["under5"], data["dhis2"],
                                                     key=f"{kn}-{ku}-{kd}-loo")
        loo = st.session_state.get("d5_loo")
        if loo is not None and not loo.empty:
            st.dataframe(loo, use_container_width=True, hide_index=True)
            ai.ai_block("d5_loo", "Zero-Dose & Hotspots - leave-one-wave-out validation",
                        "Out-of-sample accuracy of the Bayesian model: MAE/RMSE (percentage points) and "
                        "95% CI coverage when each NDHS wave is held out. Comment on whether accuracy and "
                        "interval calibration are acceptable and which wave is hardest to predict.",
                        loo.to_dict(orient="records"))
