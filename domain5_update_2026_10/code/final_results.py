"""Canonical Domain 5 results, aggregated from the executed notebook outputs (single source of truth).

Method 1  notebooks/outputs_Method1_Bayesian_Hierarchical_Model/ (773 LGAs; draws: state-model uncertainty)
Method 2  notebooks/outputs_Method2_Bayesian_Small_Area_Estimation/ (774 LGAs; full posterior draws)
Denominator for both: 12-23-month cohort = state under-5 (2024 projection) / 5, LGA share by NPC 2022 population.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
O1 = ROOT / "notebooks" / "outputs_Method1_Bayesian_Hierarchical_Model"
O2 = ROOT / "notebooks" / "outputs_Method2_Bayesian_Small_Area_Estimation"
DS = ROOT / "Updated Datasets for Domain 5"
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261009)
K20 = 155


def spearman_test(x, y, B=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = spearmanr(x, y); rg = np.random.default_rng(1)
    bs = [spearmanr(x[i], y[i]).statistic for i in (rg.integers(0, len(x), len(x)) for _ in range(B))]
    return dict(rho=float(r.statistic), p_value=float(r.pvalue), ci_lo=float(np.percentile(bs, 2.5)),
                ci_hi=float(np.percentile(bs, 97.5)), n=int(len(x)))


def within_state_test(df, a, b):
    d = df.dropna(subset=[a, b]).copy()
    d["ra"] = d.groupby("state")[a].rank(pct=True); d["rb"] = d.groupby("state")[b].rank(pct=True)
    res = spearman_test(d.ra, d.rb)
    per = d.groupby("state").apply(lambda g: spearmanr(g[a], g[b]).statistic if g[a].nunique() > 1 else np.nan,
                                   include_groups=False)
    res.update(median_state_rho=float(np.nanmedian(per)), states_positive=int((per > 0).sum()), states_tested=int(per.notna().sum()))
    return res


def k_for_share(desc):
    c = np.cumsum(desc) / np.sum(desc) * 100
    return {s: int(np.searchsorted(c, s) + 1) for s in (50, 60, 80)}


def main():
    inp = pd.read_csv(DS / "Method2_Bayesian_Small_Area_Estimation" / "sae_lga_inputs_774.csv")
    nmt = pd.read_csv(DS / "Method2_Bayesian_Small_Area_Estimation" / "nmdhs_2025_26_penta1_by_state_zone.csv")
    nms = nmt[nmt.level == "state"].set_index("area")
    nd = pd.read_csv(DS / "Method2_Bayesian_Small_Area_Estimation" / "nigeria_ndhs_zero_dose_VERIFIED_long.csv")

    # ---------------- Method 1 (notebook outputs)
    e1 = pd.read_csv(O1 / "method1_lga_estimates_2026.csv")
    s1 = pd.read_csv(O1 / "method1_state_forecasts_2026_2028.csv").set_index("state")
    z1 = np.load(O1 / "method1_lga_draws_2026.npz", allow_pickle=True)
    k1 = pd.DataFrame({"state": z1["state"], "lga": z1["lga"]})
    bd1, rt1 = z1["children"].astype(float), z1["rate"].astype(float)
    rk1 = (-bd1).argsort(1).argsort(1) + 1
    k1["m1_p_top155"] = (rk1 <= K20).mean(0)
    m1 = e1.rename(columns={"rate_2026": "m1_rate", "rate_lo95": "m1_rate_lo95", "rate_hi95": "m1_rate_hi95",
                            "children_2026": "m1_children", "children_lo95": "m1_children_lo95", "children_hi95": "m1_children_hi95",
                            "national_rank": "m1_rank", "rate_capped_99": "m1_capped_99"}).drop(columns=["p_top155", "zone", "ihme_zd_2018"])
    m1 = m1.merge(k1, on=["state", "lga"])

    # ---------------- Method 2 (notebook outputs)
    e2 = pd.read_csv(O2 / "method2_sae_lga_estimates_2026.csv")
    z2 = np.load(O2 / "method2_sae_lga_draws_2026.npz", allow_pickle=True)
    p = z2["rate"].astype(float); N = z2["cohort"].astype(float)
    k2 = pd.DataFrame({"lga_uid": z2["lga_uid"], "state": z2["state"], "lga": z2["lga"]})
    bd2 = p * N
    rk2 = (-bd2).argsort(1).argsort(1) + 1
    k2["m2_p_top155"] = (rk2 <= K20).mean(0); k2["m2_p_top270"] = (rk2 <= 270).mean(0)
    m2 = e2.rename(columns={"rate_2026": "m2_rate", "rate_lo95": "m2_rate_lo95", "rate_hi95": "m2_rate_hi95",
                            "children_2026": "m2_children", "children_lo95": "m2_children_lo95", "children_hi95": "m2_children_hi95",
                            "national_rank": "m2_rank", "rank_lo95": "m2_rank_lo95", "rank_hi95": "m2_rank_hi95",
                            "lga_cohort_12_23m_2024": "cohort_12_23m"}).drop(columns=["p_top155"])
    m2 = m2.merge(k2[["lga_uid", "m2_p_top155", "m2_p_top270"]], on="lga_uid")

    lg = m2.merge(m1, on=["state", "lga"], how="left") \
           .merge(inp[["lga_uid", "archetype", "archetype_type", "ihme_zero_dose_2018_pct", "lga_population_npc_2022"]], on="lga_uid")
    lg = lg.rename(columns={"lga": "lga_clean", "ihme_zero_dose_2018_pct": "ihme_zd_2018", "lga_population_npc_2022": "pop2022"})
    lg["m1_status"] = np.where(lg.m1_children.notna(), "estimated", "not estimated (no Penta1 data 2021-2024; partial reporting from 2025)")
    lg = lg.sort_values("m2_rank").reset_index(drop=True)
    assert len(lg) == 774 and lg.m1_children.notna().sum() == 773
    lg.to_csv(OUT / "lga_both_methods_774.csv", index=False)

    # map LGAs of draws to zone / state
    zone_of = dict(zip(zip(inp.state, inp.lga), inp.zone))
    z1zone = np.array([zone_of[(s, l)] for s, l in zip(k1.state, k1.lga)])
    z2zone = inp.set_index("lga_uid").loc[k2.lga_uid, "zone"].values
    states = sorted(inp.state.unique())

    def ci(x):
        return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    b1pt = e1.set_index(["state", "lga"]).loc[list(zip(k1.state, k1.lga)), "children_2026"].values.astype(float)
    rows = []
    for name, m1m, m2m in [("National", np.ones(len(k1), bool), np.ones(len(k2), bool))] + \
                          [(z, z1zone == z, z2zone == z) for z in sorted(inp.zone.unique())]:
        coh = N[m2m].sum()
        a1, a2 = ci(bd1[:, m1m].sum(1)), ci(bd2[:, m2m].sum(1))
        c1 = float(b1pt[m1m].sum())
        rows.append(dict(area=name, lgas_total=int(m2m.sum()), m1_lgas=int(m1m.sum()), m2_lgas=int(m2m.sum()), cohort_12_23m=float(coh),
                         m1_children=c1, m1_lo95=a1[1], m1_hi95=a1[2], m2_children=a2[0], m2_lo95=a2[1], m2_hi95=a2[2],
                         m1_rate=c1 / coh * 100, m2_rate=a2[0] / coh * 100))
    Z = pd.DataFrame(rows)
    for k in ("m1", "m2"):
        Z[f"{k}_share"] = Z[f"{k}_children"] / Z.loc[0, f"{k}_children"] * 100
    Z["difference"] = Z.m2_children - Z.m1_children; Z["difference_pct"] = Z.difference / Z.m1_children * 100
    Z.to_csv(OUT / "national_zone_both_methods.csv", index=False)

    e2s = pd.read_csv(O2 / "method2_sae_state_estimates_2026.csv").set_index("state")
    srows = []
    for s in states:
        b1 = bd1[:, (k1.state == s).values].sum(1); b2 = bd2[:, (k2.state == s).values].sum(1)
        l1 = lg[(lg.state == s) & lg.m1_rate.notna()]; l2 = lg[lg.state == s]
        srows.append(dict(state=s, zone=inp[inp.state == s].zone.iloc[0], lgas=int(len(l2)), m1_lgas=int(len(l1)),
                          ndhs_2024=float(nd[(nd.state == s) & (nd.year == 2024)].zero_dose_pct.iloc[0]),
                          nmdhs_2025_26=float(nms.loc[s, "zd_pct_100_minus_penta1"]),
                          m1_rate=float(s1.loc[s, "rate_2026"]), m1_rate_lo95=float(s1.loc[s, "rate_2026_lo95"]),
                          m1_rate_hi95=float(s1.loc[s, "rate_2026_hi95"]),
                          m1_children=float(l1.m1_children.sum()), m1_children_lo95=float(np.percentile(b1, 2.5)),
                          m1_children_hi95=float(np.percentile(b1, 97.5)),
                          m2_rate=float(e2s.loc[s, "rate_2026"]), m2_rate_lo95=float(e2s.loc[s, "rate_lo95"]), m2_rate_hi95=float(e2s.loc[s, "rate_hi95"]),
                          m2_children=float(b2.mean()), m2_children_lo95=float(np.percentile(b2, 2.5)), m2_children_hi95=float(np.percentile(b2, 97.5)),
                          cohort_12_23m=float(l2.cohort_12_23m.sum()),
                          m1_lga_rate_range=f"{l1.m1_rate.min():.0f}-{l1.m1_rate.max():.0f}",
                          m2_lga_rate_range=f"{l2.m2_rate.min():.0f}-{l2.m2_rate.max():.0f}"))
    ST = pd.DataFrame(srows)
    ST["difference"] = ST.m2_children - ST.m1_children; ST["difference_pct"] = ST.difference / ST.m1_children * 100
    ST = ST.sort_values("m2_children", ascending=False)
    ST.to_csv(OUT / "state_both_methods.csv", index=False)

    # ---------------- validation
    val = []
    for lab, col in (("Method 1", "m1_rate"), ("Method 2 (SAE)", "m2_rate")):
        e = ST[col] - ST.nmdhs_2025_26
        val.append(dict(method=lab, check="NmDHS 2025-26, 37 states (state 2026 estimate vs survey)",
                        **spearman_test(ST[col], ST.nmdhs_2025_26), MAE_pp=float(e.abs().mean()), bias_pp=float(e.mean())))
    common = lg[lg.m1_rate.notna()]
    for lab, col, sub in (("Method 1", "m1_rate", common), ("Method 2 (SAE)", "m2_rate", common), ("Method 2 (SAE), all LGAs", "m2_rate", lg)):
        d = sub.dropna(subset=["ihme_zd_2018"])
        val.append(dict(method=lab, check="IHME DTP1 2018, LGA overall", **spearman_test(d[col], d.ihme_zd_2018)))
        val.append(dict(method=lab, check="IHME DTP1 2018, LGA within state (pooled within-state ranks)", **within_state_test(sub, col, "ihme_zd_2018")))
    V = pd.DataFrame(val); V.to_csv(OUT / "validation_both_methods.csv", index=False)
    zn = nmt[nmt.level == "zone"].set_index("area")["zd_pct_100_minus_penta1"]
    Zv = Z[Z.area != "National"][["area", "m1_rate", "m2_rate"]].copy()
    Zv["m1_state_model_rate"] = [np.average(ST[ST.zone == a].m1_rate, weights=ST[ST.zone == a].cohort_12_23m) for a in Zv.area]
    Zv["nmdhs_2025_26"] = Zv.area.map(zn)
    Zv.to_csv(OUT / "zone_validation_nmdhs.csv", index=False)

    # ---------------- Pareto
    o1 = np.argsort(-b1pt)
    o2 = np.argsort(-bd2.mean(0))
    kk1 = k_for_share(np.sort(b1pt)[::-1]); kk2 = k_for_share(np.sort(bd2.mean(0))[::-1])
    par, topk = [], []
    for s in (50, 60, 80):
        sh1 = bd1[:, o1[:kk1[s]]].sum(1) / bd1.sum(1) * 100; sh2 = bd2[:, o2[:kk2[s]]].sum(1) / bd2.sum(1) * 100
        par.append(dict(share_of_burden=s, m1_lgas=kk1[s], m1_pct_of_lgas=kk1[s] / 773 * 100,
                        m1_list_share_lo95=float(np.percentile(sh1, 2.5)), m1_list_share_hi95=float(np.percentile(sh1, 97.5)),
                        m2_lgas=kk2[s], m2_pct_of_lgas=kk2[s] / 774 * 100,
                        m2_list_share_lo95=float(np.percentile(sh2, 2.5)), m2_list_share_hi95=float(np.percentile(sh2, 97.5))))
    for k in (50, 100, 155, 200, 270):
        sh1 = bd1[:, o1[:k]].sum(1) / bd1.sum(1) * 100; sh2 = bd2[:, o2[:k]].sum(1) / bd2.sum(1) * 100
        topk.append(dict(top_k=k, m1_share=float(np.sort(b1pt)[::-1][:k].sum() / b1pt.sum() * 100),
                         m1_lo95=float(np.percentile(sh1, 2.5)), m1_hi95=float(np.percentile(sh1, 97.5)),
                         m2_share=float(np.sort(bd2.mean(0))[::-1][:k].sum() / bd2.mean(0).sum() * 100),
                         m2_lo95=float(np.percentile(sh2, 2.5)), m2_hi95=float(np.percentile(sh2, 97.5)),
                         m1_near_certain=int(((rk1 <= k).mean(0) >= 0.9).sum()), m2_near_certain=int(((rk2 <= k).mean(0) >= 0.9).sum())))
    P = pd.DataFrame(par); P.to_csv(OUT / "pareto_scenarios.csv", index=False)
    pd.DataFrame(topk).to_csv(OUT / "topk_shares.csv", index=False)
    curves = pd.DataFrame({"k": np.arange(1, 775)})
    c1 = np.cumsum(np.sort(b1pt)[::-1]) / b1pt.sum() * 100
    curves["m1_cum_pct"] = np.r_[c1, [np.nan]]
    cd = np.cumsum(np.sort(bd2, 1)[:, ::-1], 1) / bd2.sum(1, keepdims=True) * 100
    curves["m2_cum_pct"] = np.cumsum(np.sort(bd2.mean(0))[::-1]) / bd2.mean(0).sum() * 100
    curves["m2_lo95"] = np.percentile(cd, 2.5, 0); curves["m2_hi95"] = np.percentile(cd, 97.5, 0)
    curves.to_csv(OUT / "pareto_curves.csv", index=False)

    # ---------------- archetypes
    A = lg.groupby(["archetype", "archetype_type"]).agg(lgas=("lga_uid", "size"), m1_lgas=("m1_children", "count"),
                                                       m1_children=("m1_children", "sum"), m2_children=("m2_children", "sum"),
                                                       m1_mean_rate=("m1_rate", "mean"), m2_mean_rate=("m2_rate", "mean"),
                                                       cohort=("cohort_12_23m", "sum")).reset_index()
    for k in ("m1", "m2"):
        A[f"{k}_share"] = A[f"{k}_children"] / A[f"{k}_children"].sum() * 100
    A["cohort_share"] = A.cohort / A.cohort.sum() * 100
    A.to_csv(OUT / "archetype_both_methods.csv", index=False)

    j1 = json.loads((O1 / "method1_summary.json").read_text()); j2 = json.loads((O2 / "method2_sae_summary.json").read_text())
    R = dict(denominator="2024 under-5 projection / 5, LGA share by NPC 2022 population",
             m1=dict(lgas=773, national=float(b1pt.sum()), national_lo95=float(np.percentile(bd1.sum(1), 2.5)),
                     national_hi95=float(np.percentile(bd1.sum(1), 97.5)), state_model_sum=float((s1.rate_2026 / 100 * s1.cohort_12_23m).sum()),
                     pareto=kk1, top155_share=float(np.sort(b1pt)[::-1][:155].sum() / b1pt.sum() * 100),
                     capped=int(lg.m1_capped_99.fillna(False).astype(bool).sum()), near_certain_top155=int((lg.m1_p_top155 >= 0.9).sum()),
                     diagnostics=j1["diagnostics"]),
             m2=dict(lgas=774, national=float(bd2.sum(1).mean()), national_lo95=float(np.percentile(bd2.sum(1), 2.5)),
                     national_hi95=float(np.percentile(bd2.sum(1), 97.5)), pareto=kk2,
                     top155_share=float(np.sort(bd2.mean(0))[::-1][:155].sum() / bd2.mean(0).sum() * 100),
                     near_certain_top155=int((lg.m2_p_top155 >= 0.9).sum()), diagnostics=j2["diagnostics"],
                     survey_ppc_coverage95=j2["survey_ppc_coverage95"]),
             overlap_top155=int(((lg.m1_rank <= 155) & (lg.m2_rank <= 155)).sum()),
             rank_corr=float(spearmanr(common.m1_children, common.m2_children).statistic))
    (OUT / "results.json").write_text(json.dumps(R, indent=2))
    pd.set_option("display.width", 250)
    print(json.dumps(R, indent=1)); print(Z.round(1).to_string()); print(V.round(3).to_string()); print(P.round(1).to_string())
    print(A.round(1).to_string()); print(Zv.round(1).to_string())


if __name__ == "__main__":
    main()
