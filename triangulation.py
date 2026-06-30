"""
Triangulation / external cross-checks: compare our model outputs against INDEPENDENT estimates.

Two convergent-validity checks:
1. Zone level - our forecast zone zero-dose vs an independent published estimate
   (Umar et al., Vaccines 2025; geometric-mean zero-dose by geopolitical zone, 2024).
2. LGA level - our 2026 LGA estimate (as DTP1 coverage) vs IHME modelled LGA DTP1 coverage
   (admin-2, 2018). Compared on rank/tercile because of the vintage gap (2026 vs 2018).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
import names as N

# Independent zone zero-dose 2024 (Umar HJ et al. Vaccines 2025;13(11):1135; geometric mean by zone).
UMAR_ZONE_2024 = {"North West": 47.4, "North Central": 26.6, "North East": 25.0,
                  "South West": 10.2, "South South": 6.2, "South East": 4.2}
UMAR_CITATION = ("Umar HJ, Onah SI, Popoola O, Jibril HH, Oyewole F. Widening geographical inequities "
                 "in DTP vaccination coverage and zero-dose prevalence across Nigeria: an ecological "
                 "trend analysis (2018-2024). Vaccines 2025;13(11):1135.")
IHME_CITATION = ("IHME Local Burden of Disease, Nigeria DTP1 vaccine coverage estimates, admin-2, "
                 "2000-2018 (GHDx, 2022).")
TERCILES = ["Lower third (worst)", "Middle third", "Upper third (best)"]


def zone_crosscheck(res: pd.DataFrame) -> dict:
    """Aggregate our 37-state forecast to zones (cohort-weighted) and compare with UMAR_ZONE_2024."""
    d = res.copy()
    d["cohort"] = pd.to_numeric(d["cohort_12_23m"], errors="coerce")
    d["zd26"] = pd.to_numeric(d["zd_count_2026"], errors="coerce")
    d["zone_k"] = d["zone"].astype(str).str.strip()
    rows = []
    for z, grp in d.groupby("zone_k"):
        coh = grp["cohort"].sum()
        if coh <= 0:
            continue
        ours26 = grp["zd26"].sum() / coh * 100
        ours24 = (grp["cohort"] * pd.to_numeric(grp["zd_obs_2024"], errors="coerce") / 100).sum() / coh * 100
        rows.append({"Zone": z, "Our model 2026 (%)": round(ours26, 1),
                     "Our model 2024 obs (%)": round(ours24, 1),
                     "Independent 2024 (%)": UMAR_ZONE_2024.get(z, np.nan)})
    tab = pd.DataFrame(rows).dropna(subset=["Independent 2024 (%)"])
    tab = tab.sort_values("Independent 2024 (%)", ascending=False).reset_index(drop=True)
    rho, p = spearmanr(tab["Our model 2026 (%)"], tab["Independent 2024 (%)"])
    return {"table": tab, "rho": round(float(rho), 3), "p": float(p), "n": len(tab)}


def load_ihme_dtp1() -> pd.DataFrame:
    """Bundled IHME admin-2 DTP1 coverage (2018), keyed for joining."""
    df = pd.read_csv(C.IHME_DTP1_ADMIN2)
    df["sk"] = df["ADM1_NAME"].map(N.nstate)
    df["lk"] = df["ADM2_NAME"].map(N.nlga)
    return df


def _match_ihme(ihme: pd.DataFrame):
    look = {(r["sk"], r["lk"]): r["dtp1_2018"] for _, r in ihme.iterrows()}
    by_state: dict = {}
    for _, r in ihme.iterrows():
        by_state.setdefault(r["sk"], []).append((str(r["lk"]).replace(" ", ""), r["dtp1_2018"]))

    def match(sk, lk):
        if (sk, lk) in look:
            return look[(sk, lk)]
        alias = N.LGA_ALIAS.get((sk, lk))
        if alias and (sk, alias) in look:
            return look[(sk, alias)]
        ns = str(lk).replace(" ", "")
        hits = [v for n, v in by_state.get(sk, [])
                if len(n) >= 5 and (ns.startswith(n) or n.startswith(ns))]
        return hits[0] if len(hits) == 1 else np.nan
    return match


def lga_crosscheck(clean_df: pd.DataFrame, ihme: pd.DataFrame) -> dict:
    """Compare our LGA zero-dose (as DTP1 coverage) with IHME LGA DTP1 (2018), by tercile."""
    o = clean_df.copy()
    o["sk"] = o["State"].map(N.nstate)
    o["lk"] = o["LGA"].map(N.nlga)
    o["our_dtp1"] = 100.0 - pd.to_numeric(o["ZD proxy (%)"], errors="coerce")
    match = _match_ihme(ihme)
    o["ihme_dtp1"] = [match(s, l) for s, l in zip(o["sk"], o["lk"])]
    m = o.dropna(subset=["ihme_dtp1", "our_dtp1"]).copy()

    rho, p = spearmanr(m["our_dtp1"], m["ihme_dtp1"])
    m["our_t"] = pd.qcut(m["our_dtp1"].rank(method="first"), 3, labels=TERCILES)
    m["ihme_t"] = pd.qcut(m["ihme_dtp1"].rank(method="first"), 3, labels=TERCILES)
    diff = (m["our_t"].cat.codes - m["ihme_t"].cat.codes).abs()
    m["Confidence"] = np.select([diff == 0, diff == 1],
                                ["High confidence", "Moderate confidence"], "Low confidence")
    ct = pd.crosstab(m["our_t"], m["ihme_t"]).reindex(index=TERCILES, columns=TERCILES).fillna(0).astype(int)
    counts = m["Confidence"].value_counts().to_dict()
    view = m[["State", "LGA", "Zone", "ZD proxy (%)", "our_dtp1", "ihme_dtp1",
              "our_t", "ihme_t", "Confidence"]].rename(columns={
                  "ZD proxy (%)": "Our zero-dose 2026 (%)", "our_dtp1": "Our DTP1 coverage 2026 (%)",
                  "ihme_dtp1": "IHME DTP1 coverage 2018 (%)", "our_t": "Our tercile",
                  "ihme_t": "IHME tercile"})
    for c in ["Our zero-dose 2026 (%)", "Our DTP1 coverage 2026 (%)", "IHME DTP1 coverage 2018 (%)"]:
        view[c] = view[c].round(1)
    view = view.sort_values("Our DTP1 coverage 2026 (%)").reset_index(drop=True)
    return {"merged": view, "rho": round(float(rho), 3), "p": float(p), "n": len(m),
            "crosstab": ct, "counts": counts,
            "n_high": int(counts.get("High confidence", 0)),
            "n_mod": int(counts.get("Moderate confidence", 0)),
            "n_low": int(counts.get("Low confidence", 0))}
