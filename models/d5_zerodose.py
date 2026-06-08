"""
Domain 5 - Zero-dose modelling.

Live Bayesian hierarchical Beta regression (PyMC) of state zero-dose rates with forecasts to
2026-2028, then a deterministic LGA burden pipeline (DHIS2 proxy -> calibrate to the state
posterior -> distribute the state cohort across LGAs by population) and Pareto prioritization.

All logic is lifted from D5__zero_dose_analysis5.ipynb (cells 14, 15, 17, 23, 24, 45).
"""
from __future__ import annotations

import difflib

import numpy as np
import pandas as pd
import streamlit as st

import config as C
import names as N
from data_io import prep_dhis2, prep_under5


# --------------------------------------------------------------------------------------
# Array preparation (D5 cells 14-15)
# --------------------------------------------------------------------------------------
def _prepare(ndhs_long: pd.DataFrame, under5: pd.DataFrame, dhis2_raw: pd.DataFrame) -> dict:
    long_df = ndhs_long.copy()
    long_df["state"] = long_df["state"].astype(str).str.strip()
    long_df["zone"] = long_df["zone"].astype(str).str.strip()

    pop = prep_under5(under5)
    pop_lookup = dict(zip(pop["jk"], pop["cohort_12_23m"]))
    pop_u5_lookup = dict(zip(pop["jk"], pop["under5_n"]))

    d = prep_dhis2(dhis2_raw)
    ann = d.groupby(["state", "year"])["penta_1_count"].sum().reset_index()
    dhis2_trend = {}
    for stt in ann["state"].unique():
        s = ann[ann["state"] == stt]
        v21 = s.loc[s["year"] == 2021, "penta_1_count"].sum()
        v24 = s.loc[s["year"] == 2024, "penta_1_count"].sum()
        dhis2_trend[stt.strip()] = (np.log(v24 / v21) / 3) if (v21 > 0 and v24 > 0) else 0.0

    year_std = long_df["year"].std()
    long_df["year_std"] = (long_df["year"] - C.YEAR_CENTER) / year_std

    states = sorted(long_df["state"].unique())
    zones = sorted(long_df["zone"].unique())
    s_idx = {s: i for i, s in enumerate(states)}
    z_idx = {z: i for i, z in enumerate(zones)}
    long_df["state_idx"] = long_df["state"].map(s_idx)

    state_zone = long_df.groupby("state")["zone"].first()
    state_zone_idx = np.array([z_idx[state_zone[s]] for s in states])

    eps = 1e-4
    long_df["zd_prop"] = (long_df["zero_dose_pct"] / 100).clip(eps, 1 - eps)

    dhis2_cov = np.array([dhis2_trend.get(s, 0.0) for s in states])
    dhis2_cov_std = (dhis2_cov - dhis2_cov.mean()) / (dhis2_cov.std() + 1e-8)

    n_obs = long_df["n_children_12_23m"].values.astype(float)
    kappa_scale = n_obs / n_obs.mean()

    fc_yr_std = np.array([(y - C.YEAR_CENTER) / year_std for y in C.FORECAST_YEARS])

    return dict(
        long_df=long_df, states=states, zones=zones, n_states=len(states), n_zones=len(zones),
        state_zone=state_zone, state_zone_idx=state_zone_idx,
        y_obs=long_df["zd_prop"].values, yr_std_v=long_df["year_std"].values,
        st_idx_v=long_df["state_idx"].values, dhis2_cov_std=dhis2_cov_std,
        kappa_scale=kappa_scale, fc_yr_std=fc_yr_std,
        pop_lookup=pop_lookup, pop_u5_lookup=pop_u5_lookup,
    )


# --------------------------------------------------------------------------------------
# Bayesian model (D5 cell 17) + state results table (D5 cell 23)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_state_model(_ndhs_long, _under5, _dhis2, key: str,
                    draws: int = C.MCMC_DRAWS_LIVE, tune: int = C.MCMC_TUNE_LIVE) -> dict:
    """Fit the hierarchical Beta model and return state results, diagnostics and metadata.

    DataFrame args are underscore-prefixed so Streamlit does not hash them (pandas-3.0
    string columns break the default hasher); caching keys on the explicit `key` string.
    """
    ndhs_long, under5, dhis2_raw = _ndhs_long, _under5, _dhis2
    import pymc as pm
    import arviz as az

    a = _prepare(ndhs_long, under5, dhis2_raw)
    n_states, n_zones = a["n_states"], a["n_zones"]
    state_zone_idx, st_idx_v = a["state_zone_idx"], a["st_idx_v"]
    yr_std_v, dhis2_cov_std, kappa_scale = a["yr_std_v"], a["dhis2_cov_std"], a["kappa_scale"]
    y_obs = a["y_obs"]

    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
        sig_a_z = pm.HalfNormal("sig_a_z", sigma=0.5)
        a_z_raw = pm.Normal("a_z_raw", mu=0, sigma=1, shape=n_zones)
        alpha_z = pm.Deterministic("alpha_z", mu_alpha + sig_a_z * a_z_raw)

        b_year_g = pm.Normal("b_year_g", mu=0, sigma=1)
        sig_b_z = pm.HalfNormal("sig_b_z", sigma=0.3)
        b_z_raw = pm.Normal("b_z_raw", mu=0, sigma=1, shape=n_zones)
        beta_z = pm.Deterministic("beta_z", b_year_g + sig_b_z * b_z_raw)

        sig_a_s = pm.HalfNormal("sig_a_s", sigma=0.5)
        z_a = pm.Normal("z_a", mu=0, sigma=1, shape=n_states)
        alpha_s = pm.Deterministic("alpha_s", alpha_z[state_zone_idx] + sig_a_s * z_a)

        sig_b_s = pm.HalfNormal("sig_b_s", sigma=0.3)
        z_b = pm.Normal("z_b", mu=0, sigma=1, shape=n_states)
        beta_s = pm.Deterministic("beta_s", beta_z[state_zone_idx] + sig_b_s * z_b)

        gamma = pm.Normal("gamma", mu=0, sigma=0.5)
        kappa_b = pm.Gamma("kappa_b", alpha=2, beta=0.5)

        eta = (alpha_s[st_idx_v] + beta_s[st_idx_v] * yr_std_v
               + gamma * dhis2_cov_std[st_idx_v])
        mu = pm.Deterministic("mu", pm.math.invlogit(eta))
        kappa_obs = kappa_b * kappa_scale
        pm.Beta("y_like", alpha=mu * kappa_obs, beta=(1 - mu) * kappa_obs, observed=y_obs)

        sample_kwargs = dict(draws=draws, tune=tune, chains=C.MCMC_CHAINS,
                             target_accept=C.TARGET_ACCEPT, random_seed=42,
                             progressbar=False, return_inferencedata=True)
        # Prefer nutpie (numba/LLVM): fast and needs no system C++ compiler. Fall back to
        # the default PyTensor sampler where a compiler is available.
        try:
            import nutpie  # noqa: F401
            trace = pm.sample(nuts_sampler="nutpie", **sample_kwargs)
        except Exception:
            trace = pm.sample(cores=1, **sample_kwargs)

    a_flat = trace.posterior["alpha_s"].values.reshape(-1, n_states)
    b_flat = trace.posterior["beta_s"].values.reshape(-1, n_states)
    g_flat = trace.posterior["gamma"].values.reshape(-1)
    fc_mu = np.zeros((a_flat.shape[0], n_states, len(C.FORECAST_YEARS)))
    for fi, fys in enumerate(a["fc_yr_std"]):
        eta_fc = a_flat + b_flat * fys + g_flat[:, None] * dhis2_cov_std[None, :]
        fc_mu[:, :, fi] = 1 / (1 + np.exp(-eta_fc))

    res = _build_state_results(a, fc_mu)

    # Diagnostics (D5 cell 19)
    diag_vars = ["mu_alpha", "b_year_g", "gamma", "kappa_b",
                 "sig_a_z", "sig_b_z", "sig_a_s", "sig_b_s"]
    summ = az.summary(trace, var_names=diag_vars, round_to=3).reset_index().rename(
        columns={"index": "Parameter"})
    # Per-variable reduction avoids broadcasting the large Deterministics into one array.
    rh = az.rhat(trace)
    es = az.ess(trace)
    max_rhat = float(max(float(rh[v].max()) for v in rh.data_vars))
    min_ess = float(min(float(es[v].min()) for v in es.data_vars))

    return {"res": res, "diag": summ, "max_rhat": round(max_rhat, 3),
            "min_ess": int(min_ess), "n_draws": a_flat.shape[0]}


def _build_state_results(a: dict, fc_mu: np.ndarray) -> pd.DataFrame:
    long_df, states, state_zone = a["long_df"], a["states"], a["state_zone"]
    rows = []
    for si, state in enumerate(states):
        sdf = long_df[long_df["state"] == state]
        jk = state.upper().replace(" ", "")
        pop = a["pop_lookup"].get(jk, np.nan)

        def obs(yr):
            v = sdf.loc[sdf["year"] == yr, "zero_dose_pct"].values
            return float(v[0]) if len(v) else np.nan

        r = dict(state=state, zone=state_zone[state],
                 pop_under5=a["pop_u5_lookup"].get(jk, np.nan), cohort_12_23m=pop,
                 zd_obs_2008=obs(2008), zd_obs_2013=obs(2013),
                 zd_obs_2018=obs(2018), zd_obs_2024=obs(2024))
        for fi, yr in enumerate(C.FORECAST_YEARS):
            draws = fc_mu[:, si, fi] * 100
            r[f"zd_pred_{yr}_mean"] = float(np.mean(draws))
            r[f"zd_pred_{yr}_lo95"] = float(np.percentile(draws, 2.5))
            r[f"zd_pred_{yr}_hi95"] = float(np.percentile(draws, 97.5))
            r[f"zd_count_{yr}"] = float(np.mean(draws)) / 100 * pop if not np.isnan(pop) else np.nan
        rows.append(r)

    res = pd.DataFrame(rows)
    res["score_rate"] = N.minmax_scale(res["zd_pred_2026_mean"])
    res["score_count"] = N.minmax_scale(res["zd_count_2026"])
    res["score_trend"] = N.minmax_scale(res["zd_obs_2024"] - res["zd_obs_2018"])
    res["score_uncert"] = N.minmax_scale(res["zd_pred_2026_hi95"] - res["zd_pred_2026_lo95"])
    res["risk_index"] = (0.45 * res["score_rate"] + 0.30 * res["score_count"]
                         + 0.15 * res["score_trend"] + 0.10 * res["score_uncert"])
    res["state_rank"] = res["risk_index"].rank(ascending=False).astype(int)
    res["priority_tier"] = pd.cut(res["risk_index"], bins=[-np.inf, 25, 50, 75, np.inf],
                                  labels=["Tier 4: Lower", "Tier 3: Moderate",
                                          "Tier 2: High", "Tier 1: Critical"])
    return res.sort_values("state_rank").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# LGA burden: even split (D5 cell 24) then population reweight + Pareto (D5 cell 45)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_lga_burden(_dhis2, _res, _lga_population, key: str) -> dict:
    dhis2_raw, res, lga_population = _dhis2, _res, _lga_population
    d = prep_dhis2(dhis2_raw)
    # DHIS2 LGA names carry a 2-letter prefix and a 'Local Government Area' suffix; clean them
    # so they display correctly and join to the population file and GRID3 geometry (D5 cell 12).
    d["lga"] = d["lga"].astype(str).map(N.clean_lga_name)
    for c in ["penta_1_count", "penta_3_count"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    grp = ["zone", "state", "lga"] if "zone" in d.columns else ["state", "lga"]
    best = []
    for yr in [2024, 2023, 2022, 2021]:
        ydf = d[d["year"] == yr].groupby(grp)[["penta_1_count", "penta_3_count"]].sum().reset_index()
        ydf["yr"] = yr
        best.append(ydf)
    lga_all = pd.concat(best, ignore_index=True)
    lga_all = lga_all[lga_all["penta_1_count"] > 0]
    lga_best = (lga_all.sort_values(["state", "lga", "yr"], ascending=[True, True, False])
                .drop_duplicates(subset=["state", "lga"], keep="first").reset_index(drop=True))

    state_pred = dict(zip(res["state"], res["zd_pred_2026_mean"] / 100))
    state_cohort = dict(zip(res["state"], res["cohort_12_23m"]))

    out_rows = []
    for state, sg in lga_best.groupby("state"):
        sc = str(state).strip()
        zd = state_pred.get(sc) or state_pred.get(sc.title()) or state_pred.get(N.normalise_name(sc))
        pop = state_cohort.get(sc) or state_cohort.get(sc.title()) or state_cohort.get(N.normalise_name(sc))
        if zd is None or (isinstance(zd, float) and np.isnan(zd)):
            continue
        sg = sg.copy()
        tot = sg["penta_1_count"].sum()
        if tot == 0:
            continue
        sg["p1_share"] = sg["penta_1_count"] / tot
        sg["dropout_p1p3"] = ((sg["penta_1_count"] - sg["penta_3_count"])
                              / sg["penta_1_count"].replace(0, np.nan) * 100).clip(-50, 100)
        mean_share = sg["p1_share"].mean()
        sg["zd_proxy_raw"] = zd * (1 + (mean_share - sg["p1_share"]) / (mean_share + 1e-6))
        sg["zd_proxy_pct"] = (sg["zd_proxy_raw"] * 100).clip(1, 99)
        if pop is not None and not (isinstance(pop, float) and np.isnan(pop)):
            sg["lga_pop_est"] = float(pop) / len(sg)
            sg["zd_count_est"] = (sg["zd_proxy_pct"] / 100 * sg["lga_pop_est"]).round(0)
        else:
            sg["lga_pop_est"] = np.nan
            sg["zd_count_est"] = np.nan
        out_rows.append(sg)

    if not out_rows:
        return {"clean": pd.DataFrame(), "pareto": pd.DataFrame(), "national_total": 0,
                "n_lgas": 0, "top20_pct": 0, "n80": 0, "matched_pop": 0}

    lga = pd.concat(out_rows, ignore_index=True)
    lga["score_rate"] = lga.groupby("state")["zd_proxy_pct"].transform(N.minmax_scale)
    lga["score_dropout"] = lga.groupby("state")["dropout_p1p3"].transform(
        lambda x: N.minmax_scale(x.fillna(x.median())))
    lga["lga_risk_index"] = 0.60 * lga["score_rate"] + 0.40 * lga["score_dropout"]
    lga["lga_tier"] = pd.cut(lga["lga_risk_index"], bins=[-np.inf, 25, 50, 75, np.inf],
                             labels=["Tier 4", "Tier 3", "Tier 2", "Tier 1"])

    clean, pareto, stats = _population_weight(lga, lga_population)
    return {"clean": clean, "pareto": pareto, **stats}


def _population_weight(lga: pd.DataFrame, pop_raw: pd.DataFrame) -> tuple:
    """Distribute each state cohort across LGAs by population (D5 cell 45)."""
    pop = pop_raw.copy()
    pop = pop[pop["Status"].astype(str).str.strip() == "Local Government Area"].copy()
    pop.loc[pop["Name"].isin(N.FCT6), "State"] = "FCT"
    pop["ns"] = pop["State"].map(N.nstate)
    pop["nl"] = pop["Name"].map(N.nlga)
    pop["tk"] = pop["Name"].map(N.tok)
    pop["P"] = pd.to_numeric(pop["PopulationProjection2022-03-21"], errors="coerce")

    def match(st, nl, tk):
        nl = N.LGA_ALIAS.get((st, nl), nl)
        sub = pop[pop["ns"] == st]
        for col, val in [("nl", nl), ("tk", tk)]:
            m = sub[sub[col] == val]
            if len(m):
                return float(m["P"].iloc[0])
        c = difflib.get_close_matches(nl, list(sub["nl"]), n=1, cutoff=0.80)
        return float(sub[sub["nl"] == c[0]]["P"].iloc[0]) if c else np.nan

    cl = lga.copy()
    cl["LGA population (2022)"] = [match(N.nstate(s), N.nlga(l), N.tok(l))
                                   for s, l in zip(cl["state"], cl["lga"])]
    matched = int(cl["LGA population (2022)"].notna().sum())
    cl["LGA population (2022)"] = cl.groupby("state")["LGA population (2022)"].transform(
        lambda s: s.fillna(s.mean() if s.notna().any() else 1.0))

    rate = pd.to_numeric(cl["zd_proxy_pct"], errors="coerce") / 100.0
    cur = pd.to_numeric(cl["zd_count_est"], errors="coerce")  # even-split state totals to preserve
    raw = rate * cl["LGA population (2022)"]
    scur = cur.groupby(cl["state"]).transform("sum")
    sraw = raw.groupby(cl["state"]).transform("sum")
    cl["zd_count_w"] = (raw * scur / sraw).round(0)
    cl["zd_count_w"] = cl["zd_count_w"].fillna(0).astype(int)
    cl["LGA population (2022)"] = cl["LGA population (2022)"].round(0).astype(int)

    sevmap = {"Tier 1": "Critical", "Tier 2": "High", "Tier 3": "Moderate", "Tier 4": "Lower"}
    cl["Severity (within state)"] = cl["lga_tier"].astype(str).str.strip().map(sevmap)

    cl = cl.sort_values("zd_count_w", ascending=False, kind="stable").reset_index(drop=True)
    cl["National rank"] = np.arange(1, len(cl) + 1)
    cl["State rank"] = cl.groupby("state")["zd_count_w"].rank(ascending=False, method="first").astype(int)

    clean = cl.rename(columns={
        "zone": "Zone", "state": "State", "lga": "LGA",
        "penta_1_count": "Penta1 count", "penta_3_count": "Penta3 count",
        "p1_share": "Penta1 share", "zd_proxy_pct": "ZD proxy (%)",
        "zd_count_w": "ZD count (est)", "dropout_p1p3": "Dropout P1-P3 (%)",
        "lga_risk_index": "LGA risk index", "lga_tier": "LGA tier",
    })
    order = ["National rank", "State rank", "Zone", "State", "LGA", "Penta1 count",
             "Penta3 count", "Penta1 share", "ZD proxy (%)", "LGA population (2022)",
             "ZD count (est)", "Dropout P1-P3 (%)", "LGA risk index", "LGA tier",
             "Severity (within state)"]
    clean = clean[[c for c in order if c in clean.columns]]
    for c in ["Penta1 share", "ZD proxy (%)", "Dropout P1-P3 (%)", "LGA risk index"]:
        if c in clean:
            clean[c] = pd.to_numeric(clean[c], errors="coerce").round(2)

    total = int(clean["ZD count (est)"].sum())
    par = clean.sort_values("ZD count (est)", ascending=False).reset_index(drop=True)
    pareto = pd.DataFrame({
        "Burden rank": np.arange(1, len(par) + 1), "State": par["State"], "LGA": par["LGA"],
        "Zone": par["Zone"], "LGA population (2022)": par["LGA population (2022)"],
        "Zero-dose children (est)": par["ZD count (est)"],
        "Zero-dose rate (%)": pd.to_numeric(par["ZD proxy (%)"]).round(2),
        "LGA tier": par["LGA tier"], "Severity (within state)": par["Severity (within state)"],
    })
    cum = pareto["Zero-dose children (est)"].cumsum() / total * 100
    pareto["Cumulative % of burden"] = cum.round(1)
    pareto["Priority band"] = np.where(cum <= 50, "A: drives first 50%",
                                       np.where(cum <= 80, "B: 50-80%", "C: long tail 80-100%"))
    n20 = int(round(0.20 * len(pareto)))
    pareto["Top 20% LGA"] = np.where(pareto["Burden rank"] <= n20, "Yes", "No")

    n80 = int((cum <= 80).sum() + 1)
    top20_pct = float(cum.iloc[n20 - 1]) if n20 >= 1 else 0.0
    stats = {"national_total": total, "n_lgas": len(clean), "top20_pct": round(top20_pct, 1),
             "n80": n80, "matched_pop": matched}
    return clean, pareto, stats
