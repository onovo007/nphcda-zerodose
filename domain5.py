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
from models.d5_zerodose import run_state_model, run_lga_burden


def _download(df: pd.DataFrame, label: str, fname: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), fname, "text/csv")


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
    kpi_row([
        {"label": "National zero-dose, 2026", "value": f"{nat_2026/1e6:.2f}M",
         "sub": "state-model sum", "color": C.ACCENT},
        {"label": "LGA-sum burden", "value": f"{lga['national_total']:,}",
         "sub": f"across {lga['n_lgas']} LGAs", "color": C.NAVY},
        {"label": "Highest-risk state", "value": clean(top_state["state"]),
         "sub": f"{top_state['zd_pred_2026_mean']:.0f}% predicted 2026", "color": C.GOLD},
        {"label": "Pareto concentration", "value": f"top 20% = {lga['top20_pct']:.0f}%",
         "sub": f"80% of burden in {lga['n80']} LGAs", "color": C.STEEL},
        {"label": "Convergence", "value": f"R-hat {out['max_rhat']}",
         "sub": f"min ESS {out['min_ess']} | {out['n_draws']} draws", "color": C.STEEL},
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
        col_sev = "Severity (within state)" if "Severity (within state)" in pareto.columns else None
        show = highlight_classes(pareto.head(60), col_sev, SEVERITY_CELL) if col_sev else pareto.head(60)
        st.dataframe(show, use_container_width=True, height=420)
        _download(pareto, "Download Pareto priority table (CSV)", "D5_lga_pareto_priority.csv")
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
                         "Moderate (amber), Tier 4 Lower (blue). Act on Tier 1 LGAs first."))
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
