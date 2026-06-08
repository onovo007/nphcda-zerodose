"""Domain 5 view - Zero-dose modelling and hotspots."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config as C
import names as N
import viz
import ai
import data_io as io
from theme import section, kpi_row, clean
from models.d5_zerodose import run_state_model, run_lga_burden


def _download(df: pd.DataFrame, label: str, fname: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), fname, "text/csv")


def render(data: dict):
    st.markdown("## Domain 5 - Zero-dose modelling and hotspot detection")
    st.caption(clean("Research question: where are zero-dose children most concentrated, and "
                     "what local factors contribute? Bayesian hierarchical Beta regression of "
                     "state rates, distributed to LGA burden by population, with Getis-Ord Gi* hotspots."))

    needed = {"ndhs_long", "under5", "dhis2", "lga_population"}
    if not data or any(data.get(k) is None for k in needed):
        st.warning("Domain 5 needs the NDHS longitudinal, under-five population, DHIS2 and LGA "
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
        ai.ai_block("d5_state", "Domain 5 - state zero-dose forecasts and risk ranking",
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
        st.dataframe(pareto.head(60), use_container_width=True, height=420)
        _download(pareto, "Download Pareto priority table (CSV)", "D5_lga_pareto_priority.csv")
        ai.ai_block("d5_pareto", "Domain 5 - LGA Pareto concentration of zero-dose burden",
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
                "LGA-level local spatial autocorrelation (k=5 nearest neighbours) and estimated rate.")
        import spatial
        gi = spatial.lga_gi_star(clean_df.rename(columns={
            "State": "state", "LGA": "lga", "ZD proxy (%)": "zd_proxy_pct"})[["state", "lga", "zd_proxy_pct"]],
            key=lkey)
        gdf = spatial.load_gdf("lga").merge(
            gi[["state_key", "lga_key", "zd_proxy_pct", "gi_class"]],
            on=["state_key", "lga_key"], how="left")
        gdf["gi_class"] = gdf["gi_class"].fillna("Not Significant")
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
        ai.ai_block("d5_hot", "Domain 5 - Getis-Ord Gi* hotspot clusters",
                    "Counts of LGAs in each Gi* category (statistically significant hot spots and "
                    "cold spots of zero-dose burden, k=5 nearest neighbours), and the strongest "
                    "hotspot LGAs.", hot_ctx)
        st.divider()
        ai.chat_panel("d5_hot_chat", "Gi* hotspot clusters",
                      "Counts of LGAs per Gi* category and the strongest hotspot LGAs by z-score.",
                      hot_ctx,
                      suggestions=["Which LGAs are the strongest hotspots?", "How many hot spots at p<0.01?"])

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
        st.dataframe(view, use_container_width=True, height=480)
        _download(clean_df, "Download full ranked LGA table (CSV)", "D5_lga_zero_dose_ranked_CLEAN.csv")
        ai.ai_block("d5_lga", "Domain 5 - ranked LGA table (population-weighted)",
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
