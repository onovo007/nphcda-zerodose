"""Assemble 'Updated Datasets for Domain 5': self-contained inputs for the three Colab notebooks."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent / "worldbank_lsms_2026_domain5_20261008"
sys.path.insert(0, str(WS / "05_code"))
import d5config as C  # noqa: E402

AF = C.ARCH_FILES / "AIN"
D = ROOT / "Updated Datasets for Domain 5"
M1, M2, AR = D / "Method1_Bayesian_Hierarchical_Model", D / "Method2_Bayesian_Small_Area_Estimation", D / "LGA_Archetypes"
for d in (M1, M2, AR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- shared sources
ndhs = AF / "claude_model/data/sample/nigeria_ndhs_zero_dose_VERIFIED_long.csv"
u24, u25 = AF / "claude_model/data/sample/under_5_2024.csv", AF / "claude_model/data/sample/under_5_2025.csv"
geo_l, geo_s = AF / "claude_model/data/sample/geo/nga_lgas.geojson", AF / "claude_model/data/sample/geo/nga_states.geojson"
nm = pd.read_parquet(C.D["ext"] / "domain5_nmdhs_2025_26_table10_parsed.parquet")
nm = nm.rename(columns={"name": "area"})[["level", "area", "zone", "penta1_pct", "penta3_pct", "no_vacc_pct",
                                         "n_weighted_12_23m", "small_n_flag_25_49", "zd_pct_100_minus_penta1"]]
ihme = AF / "claude_model/data/sample/ihme_dtp1_admin2_2018.csv"

# ---------------- Method 1
for src in (ndhs, u24, u25, AF / "claude_model/data/sample/dhis2_data_all_states.csv",
            AF / "claude_model/data/sample/administrative_lga_population.csv", geo_l, geo_s, ihme):
    shutil.copy2(src, M1 / src.name)
nm.to_csv(M1 / "nmdhs_2025_26_penta1_by_state_zone.csv", index=False)

# ---------------- Method 2 (SAE)
sae = pd.read_csv(WS / "16_SAE_joint_bayesian_model/inputs/sae_lga_inputs_774.csv")
cov = pd.read_parquet(C.D["integ"] / "domain5_all774_covariates_and_missingness.parquet")
from ihme_join import ihme_zero_dose  # noqa: E402
ih = sae[["state", "lga_clean"]].copy()
ih["ihme_zd_2018"] = ihme_zero_dose(ihme, ih.state, ih.lga_clean)
raw = ["anc4plus", "delivery_hf", "improved_water", "relative_wealth_index", "travel_time_hc", "conflict_events", "poverty_rate"]
t = sae[["lga_uid", "zone", "state", "lga_clean", "state_key", "geo_lga_key", "pop2022", "pop_share_in_state", "state_cohort",
         "N_cohort", "p1_annual", "p1_months_pos", "admin_coverage_pct"]].merge(cov[["state", "lga_clean"] + raw + ["archetype", "archetype_type"]],
                                                                         on=["state", "lga_clean"], how="left").merge(ih, on=["state", "lga_clean"], how="left")
t = t.rename(columns={"lga_clean": "lga", "state_cohort": "state_cohort_12_23m_2024", "N_cohort": "lga_cohort_12_23m_2024",
                      "pop2022": "lga_population_npc_2022", "p1_annual": "dhis2_penta1_annual_mean_2023_25",
                      "p1_months_pos": "dhis2_penta1_months_reported_2023_25", "admin_coverage_pct": "dhis2_admin_penta1_coverage_pct",
                      "ihme_zd_2018": "ihme_zero_dose_2018_pct"})
assert len(t) == 774 and t.lga_uid.is_unique
t.to_csv(M2 / "sae_lga_inputs_774.csv", index=False)
e = pd.read_csv(WS / "16_SAE_joint_bayesian_model/inputs/sae_adjacency_edges.csv")
uid = t.lga_uid.values
pd.DataFrame({"lga_uid_1": uid[e.i], "lga_uid_2": uid[e.j]}).to_csv(M2 / "lga_adjacency_grid3_queen.csv", index=False)
for src in (ndhs, u24, u25, geo_l, geo_s):
    shutil.copy2(src, M2 / src.name)
nm.to_csv(M2 / "nmdhs_2025_26_penta1_by_state_zone.csv", index=False)

# ---------------- Archetypes
src = pd.read_excel(AF / "Archtyping at LGA level/LGA_archetype_master.xlsx", "Sources")
covs15 = list(src["covariate"])
m = cov[["lga_uid", "zone", "state", "lga_clean", "platform_State", "platform_LGA", "archetype", "archetype_type"] + covs15]
assert m.lga_uid.notna().all() and len(m) == 774 and m.archetype.notna().all()
m.rename(columns={"lga_clean": "lga"}).to_csv(AR / "lga_archetype_covariates_774.csv", index=False)
src.to_csv(AR / "archetype_covariate_sources.csv", index=False)
est = pd.read_csv(ROOT / "results" / "lga_both_methods_774.csv")
est = est[["lga_uid", "zone", "state", "lga_clean", "cohort_12_23m", "m1_rate", "m1_children", "m2_rate", "m2_rate_lo95",
           "m2_rate_hi95", "m2_children", "m2_children_lo95", "m2_children_hi95"]].rename(columns={
    "lga_clean": "lga", "m1_rate": "method1_zero_dose_rate_pct", "m1_children": "method1_zero_dose_children",
    "m2_rate": "method2_zero_dose_rate_pct", "m2_rate_lo95": "method2_rate_lo95", "m2_rate_hi95": "method2_rate_hi95",
    "m2_children": "method2_zero_dose_children", "m2_children_lo95": "method2_children_lo95", "m2_children_hi95": "method2_children_hi95"})
est.to_csv(AR / "lga_zero_dose_estimates_2026_both_methods.csv", index=False)
xw = pd.read_parquet(C.D["integ"] / "domain5_all774_lga_geographic_crosswalk.parquet")
xw["geo_lga_key"] = np.where(xw.grid3_match == "exact", xw.lga_key, xw.grid3_match.str.split(":", n=1).str[-1])
xw[["lga_uid", "state", "lga_clean", "state_key", "geo_lga_key"]].rename(columns={"lga_clean": "lga"}).to_csv(AR / "lga_map_keys_774.csv", index=False)
for s_ in (geo_l, geo_s):
    shutil.copy2(s_, AR / s_.name)
print({d.name: sorted(p.name for p in d.iterdir()) for d in (M1, M2, AR)})
