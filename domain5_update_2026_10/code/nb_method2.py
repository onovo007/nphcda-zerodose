"""Build notebook: Domain 5 Method 2, Bayesian small-area estimation (SAE), written for newcomers to SAE."""
import nbformat as nbf

from nb_common import NAMES_CODE, code, md, setup_cell

REQ = ["sae_lga_inputs_774.csv", "lga_adjacency_grid3_queen.csv", "nigeria_ndhs_zero_dose_VERIFIED_long.csv",
       "under_5_2024.csv", "nmdhs_2025_26_penta1_by_state_zone.csv", "nga_lgas.geojson", "nga_states.geojson"]
PIP = ["pymc", "nutpie", "arviz", "geopandas", "openpyxl"]

cells = [
md("""
# Domain 5, Method 2: Bayesian small-area estimation (SAE) of zero-dose children for all 774 LGAs

**National Primary Health Care Development Agency (NPHCDA) - Predictive Modelling of Zero-Dose Children.**
Prepared by the CIDRE and Quantium Insights LLC consortium, with UNICEF and Gavi.

### The problem small-area estimation solves
Household surveys such as the Nigeria Demographic and Health Survey (NDHS) measure zero-dose prevalence reliably for **states**, but they interview too few households in each local government area (LGA) to give a direct LGA estimate. Programmes, however, plan and deliver at LGA level. **Small-area estimation (SAE)** combines the reliable state survey results with information that *is* available for every LGA (local conditions and geography) to produce LGA estimates with honest uncertainty.

### The idea in three sentences
1. Each LGA has its own unknown zero-dose rate.
2. The population-weighted average of the LGA rates in a state must agree with that state's survey results, allowing for survey sampling error. **The survey sets the level.**
3. How rates differ *between LGAs inside a state* is learned from local conditions (maternal care, water, wealth, travel time, conflict, poverty) and from neighbouring LGAs, which tend to be alike.

Because the model is Bayesian, every LGA estimate comes with a **credible interval** (a range that contains the true value with 95% probability, given the model and data) and a **probability of being a priority LGA**.

**How to run:** Runtime > Run all, then upload the files from the folder `Method2_Bayesian_Small_Area_Estimation` of *Updated Datasets for Domain 5*. Model fitting takes about 5-10 minutes on Colab.
"""),
setup_cell(REQ, PIP),
code(NAMES_CODE),
md("""
## 1. The data

| File | One row per | What it gives the model |
|---|---|---|
| `nigeria_ndhs_zero_dose_VERIFIED_long.csv` | state and survey round (37 x 4 = 148) | **Outcome.** NDHS zero-dose % (100 minus DTP1 coverage) for 2008, 2013, 2018 and 2023-24 (coded 2024), and the number of children in the survey table. |
| `sae_lga_inputs_774.csv` | LGA (774) | Population share within the state, the 12-23-month cohort, and the raw values of the local-condition covariates. |
| `lga_adjacency_grid3_queen.csv` | pair of neighbouring LGAs (2,165) | Which LGAs share a border (GRID3 boundaries); used for spatial smoothing. |
| `under_5_2024.csv` | state | Under-five population, 2024 projection (cohort = under-five / 5). |
| `nmdhs_2025_26_...csv` | state, zone | **Independent check only** (not used to fit the model). |
"""),
code(r'''
lga = pd.read_csv(DATA_DIR / "sae_lga_inputs_774.csv").sort_values(["state", "lga"]).reset_index(drop=True)
ndhs = pd.read_csv(DATA_DIR / "nigeria_ndhs_zero_dose_VERIFIED_long.csv")
edges_uid = pd.read_csv(DATA_DIR / "lga_adjacency_grid3_queen.csv")
print("LGAs:", len(lga), "| states:", lga.state.nunique(), "| survey state-rounds:", len(ndhs), "| neighbour pairs:", len(edges_uid))
assert len(lga) == 774 and lga.lga_uid.is_unique and len(ndhs) == 148

# Cohort check: LGA cohort = state under-five (2024) / 5 x LGA share of the state's NPC 2022 population
u = pd.read_csv(DATA_DIR / "under_5_2024.csv").iloc[1:]; u.columns = ["zone_abbr", "state_raw", "under5"]
u["cohort"] = (u.under5.astype(str).str.replace(",", "").astype(float) / 5).round(0)
u["jk"] = u.state_raw.str.strip().str.upper().str.replace(" ", "").str.replace(",ABUJA", "").str.replace(",", "")
coh = dict(zip(u.jk, u.cohort))
chk = lga.state.map(lambda s: coh[s.upper().replace(" ", "")]) * lga.pop_share_in_state
assert np.allclose(chk, lga.lga_cohort_12_23m_2024), "cohort mismatch"
print("Children aged 12-23 months (2024 projection):", f"{lga.lga_cohort_12_23m_2024.sum():,.0f}")
lga.head(3)
'''),
md("""
## 2. Preparing the local-condition covariates
Six covariates describe each LGA. None of them is a vaccination measure.

| Covariate | Source | Expected direction |
|---|---|---|
| Maternal-care index = average of standardised ANC4+ and facility delivery (they are 92% correlated, so they are combined) | DHS Spatial Data Repository 2018 | more care, fewer zero-dose |
| Improved water source (%) | DHS Spatial Data Repository 2018 | more access, fewer zero-dose |
| Relative Wealth Index | Meta Data for Good | wealthier, fewer zero-dose |
| log(1 + travel time to nearest facility, minutes) | Weiss et al. 2020 | farther, more zero-dose |
| log(1 + conflict events 2021-2024) | ACLED | more conflict, more zero-dose |
| Poverty rate (%) | poverty surface | poorer, more zero-dose |

Seven LGAs lack the DHS surfaces; they receive their state's median. All covariates are then **standardised** (mean 0, standard deviation 1), so each effect is "per standard deviation".
"""),
code(r'''
z = lambda s: (s - s.mean()) / s.std()
X = pd.DataFrame(index=lga.index)
X["maternal_care_index"] = (z(lga.anc4plus) + z(lga.delivery_hf)) / 2
X["improved_water"] = lga.improved_water
X["relative_wealth_index"] = lga.relative_wealth_index
X["log_travel_time"] = np.log1p(lga.travel_time_hc)
X["log_conflict_events"] = np.log1p(lga.conflict_events)
X["poverty_rate"] = lga.poverty_rate
imputed = X.isna().any(axis=1)
X = X.groupby(lga.state).transform(lambda s: s.fillna(s.median())).fillna(X.median())
X = (X - X.mean()) / X.std()
COVS = list(X.columns)
print("LGAs with an imputed covariate:", int(imputed.sum()))
X.describe().round(2)
'''),
md(r"""
## 3. Neighbours and the spatial (BYM2) effect
Neighbouring LGAs often share conditions that no covariate captures (for example, a shared health facility catchment or insecurity). The **BYM2** model (Besag-York-Mollie, version 2; Riebler et al., 2016) gives each LGA a residual effect made of two parts:
* a **structured** part $\phi$ that is similar between neighbours (an intrinsic conditional autoregressive, ICAR, effect), and
* an **unstructured** part $\theta$ that is independent for each LGA.

$$c_l = \sigma_u\left(\sqrt{\rho/s}\;\phi_l + \sqrt{1-\rho}\;\theta_l\right)$$

$\sigma_u$ is the overall size of residual variation, $\rho$ is the share that is spatially structured, and $s$ is a **scaling factor** computed from the neighbour graph so that $\sigma_u$ has the same meaning on any map. The ICAR part is implemented as a penalty on differences between neighbours, $-\tfrac12\sum_{l\sim k}(\phi_l-\phi_k)^2$, plus a constraint that the $\phi$ sum to (almost) zero.
"""),
code(r'''
pos = {u_: i for i, u_ in enumerate(lga.lga_uid)}
edges = np.array([[pos[a], pos[b]] for a, b in zip(edges_uid.lga_uid_1, edges_uid.lga_uid_2)])
n = len(lga)
W = np.zeros((n, n)); W[edges[:, 0], edges[:, 1]] = 1; W[edges[:, 1], edges[:, 0]] = 1
from scipy.sparse.csgraph import connected_components
assert connected_components(W)[0] == 1, "neighbour graph must be connected"
Q = np.diag(W.sum(1)) - W                         # ICAR precision (graph Laplacian)
lam, V = np.linalg.eigh(Q); keep = lam > 1e-9
Q_inv = (V[:, keep] / lam[keep]) @ V[:, keep].T   # generalised inverse on the sum-to-zero space
scale = float(np.exp(np.mean(np.log(np.diag(Q_inv)))))
print(f"Neighbour pairs: {len(edges)} | average neighbours per LGA: {W.sum(1).mean():.1f} | BYM2 scaling factor: {scale:.3f}")
'''),
md(r"""
## 4. The model

**LGA rate.** For LGA $l$ in state $s$ in year $y$ (time standardised as $t_y$):
$$\mathrm{logit}\,p_{l,y} = \alpha_s + \beta_s t_y + u_l, \qquad u_l = X_l b + c_l - \textstyle\sum_{k\in s} w_k c_k$$
* $\alpha_s, \beta_s$: the state's level and time trend, partially pooled within zones and then nationally (states with weak data borrow from their zone).
* $X_l b$: covariate effects. They are learned from how covariates differ **between states** and relate to the surveys, then applied within states.
* $c_l$: BYM2 residual, centred within each state (its weighted average is zero) so it only redistributes children inside a state.

**Survey link (the anchor).** The state's prevalence is the cohort-weighted mean of its LGA rates, $P_{s,y} = \sum_{l\in s} w_l\, p_{l,y}$ with $w_l$ = LGA share of the state's 12-23-month cohort. Each published NDHS value is treated as a noisy measurement of it:
$$\mathrm{logit}(\hat p_{s,y}) \sim N\!\left(\mathrm{logit}\,P_{s,y},\ \sqrt{v_{s,y}+\tau^2}\right), \qquad v_{s,y} = \frac{\mathrm{deff}}{n_{s,y}\,\tilde p(1-\tilde p)}$$
$v$ is the **sampling variance** of a survey proportion on the logit scale (design effect deff = 2 for cluster sampling; $\tilde p$ adds a small continuity correction), and $\tau$ allows true year-to-year deviation from the smooth trend.

**Priors** (weakly informative): national mean and trend $N(0,1)$; zone effects sum to zero with scales Half-Normal(0.5) and (0.3); state scales Half-Normal(0.5) and (0.3); $\tau$ and $\sigma_u$ Half-Normal(0.5) and (1); $\rho$ Beta(0.5, 0.5); covariate effects $N(0, 0.5)$.
"""),
code(r'''
import pymc as pm, pytensor.tensor as pt, arviz as az

states = sorted(lga.state.unique()); zones = sorted(lga.zone.unique())
s_idx = lga.state.map({s: i for i, s in enumerate(states)}).values
st_zone = lga.groupby("state").zone.first().loc[states]
sz_idx = st_zone.map({z_: i for i, z_ in enumerate(zones)}).values
w = (lga.lga_cohort_12_23m_2024 / lga.groupby("state").lga_cohort_12_23m_2024.transform("sum")).values
M = np.zeros((len(states), n)); M[s_idx, np.arange(n)] = w            # state = weighted mean of its LGAs

YEAR_CENTER = 2024
years = sorted(ndhs.year.unique()); ysd = ndhs.year.std()
t_w = (np.array(years) - YEAR_CENTER) / ysd
obs = ndhs.assign(si=ndhs.state.map({s: i for i, s in enumerate(states)}), wi=ndhs.year.map({y: i for i, y in enumerate(years)}))
DEFF = 2.0
n_obs = obs.n_children_12_23m.values.astype(float)
p_tilde = (obs.zero_dose_pct.values / 100 * n_obs + 0.5) / (n_obs + 1)
v = DEFF / (n_obs * p_tilde * (1 - p_tilde))                          # sampling variance on the logit scale
y_logit = np.log(p_tilde / (1 - p_tilde))
Xv = X.values

with pm.Model(coords={"state": states, "zone": zones, "cov": COVS}) as sae:
    # state hierarchy: national -> zone -> state, for level (alpha) and trend (beta)
    mu_alpha = pm.Normal("mu_alpha", 0, 1)
    alpha_z = mu_alpha + pm.HalfNormal("sig_a_z", 0.5) * pm.ZeroSumNormal("a_z_raw", sigma=1, dims="zone")
    b_year_g = pm.Normal("b_year_g", 0, 1)
    beta_z = b_year_g + pm.HalfNormal("sig_b_z", 0.3) * pm.ZeroSumNormal("b_z_raw", sigma=1, dims="zone")
    alpha_s = pm.Deterministic("alpha_s", alpha_z[sz_idx] + pm.HalfNormal("sig_a_s", 0.5) * pm.Normal("z_a", 0, 1, dims="state"), dims="state")
    beta_s = pm.Deterministic("beta_s", beta_z[sz_idx] + pm.HalfNormal("sig_b_s", 0.3) * pm.Normal("z_b", 0, 1, dims="state"), dims="state")
    tau = pm.HalfNormal("tau", 0.5)
    # BYM2 spatial residual
    sigma_u = pm.HalfNormal("sigma_u", 1.0)
    rho = pm.Beta("rho_bym2", 0.5, 0.5)
    phi = pm.Normal("phi_icar", 0, 1, shape=n)
    pm.Potential("icar", -0.5 * pt.sum((phi[edges[:, 0]] - phi[edges[:, 1]]) ** 2))
    pm.Potential("icar_sum0", pm.logp(pm.Normal.dist(0, 0.001 * n), pt.sum(phi)))
    theta = pm.Normal("theta_iid", 0, 1, shape=n)
    c = sigma_u * (pt.sqrt(rho / scale) * phi + pt.sqrt(1 - rho) * theta)
    # covariates (uncentred) + spatial residual centred within each state
    b = pm.Normal("b_cov", 0, 0.5, dims="cov")
    u_l = pm.Deterministic("u_lga", pt.dot(Xv, b) + c - pt.dot(M, c)[s_idx])
    # survey likelihood on the aggregated LGA rates
    eta = alpha_s[s_idx][:, None] + beta_s[s_idx][:, None] * t_w[None, :] + u_l[:, None]   # LGA x survey round
    P = pt.dot(M, pm.math.invlogit(eta))                                                   # state x survey round
    P_obs = pt.clip(P[obs.si.values, obs.wi.values], 1e-6, 1 - 1e-6)
    pm.Normal("y_survey", mu=pt.log(P_obs / (1 - P_obs)), sigma=pt.sqrt(v + tau ** 2), observed=y_logit)
print("Model built:", len(sae.free_RVs), "parameter blocks")
'''),
md("""
## 5. Fitting the model (Markov chain Monte Carlo)
The posterior is explored with the No-U-Turn Sampler (NUTS): 4 independent chains, 3,000 tuning steps and 8,000 kept draws each. Long chains are used because the spatial variance parameters mix slowly.
"""),
code(r'''
kw = dict(draws=8000, tune=3000, chains=4, target_accept=0.95, random_seed=2030, progressbar=False)
with sae:
    try:
        trace = pm.sample(nuts_sampler="nutpie", **kw)
    except Exception as e:
        print("nutpie unavailable, using the default PyMC sampler:", e)
        trace = pm.sample(**kw)
'''),
md("""
## 6. Did the sampler converge?
* **R-hat** compares the four chains; values at or below 1.01 for every parameter mean they agree.
* **Effective sample size (ESS)** is the number of independent draws the chains are worth; several hundred or more is ample for 95% intervals.
* **Divergent transitions** flag regions the sampler could not explore; there should be none.
"""),
code(r'''
names = ["mu_alpha", "sig_a_z", "a_z_raw", "b_year_g", "sig_b_z", "b_z_raw", "sig_a_s", "sig_b_s", "z_a", "z_b", "tau",
         "sigma_u", "rho_bym2", "b_cov", "phi_icar", "theta_iid"]
summ = az.summary(trace, var_names=names)
diag = dict(parameters=len(summ), max_rhat=float(pd.to_numeric(summ.r_hat).max()), min_ess_bulk=float(pd.to_numeric(summ.ess_bulk).min()),
            min_ess_tail=float(pd.to_numeric(summ.ess_tail).min()), divergences=int(trace.sample_stats["diverging"].sum()))
print(diag)
assert diag["max_rhat"] <= 1.01 and diag["divergences"] == 0, "not converged: lengthen the chains"
az.summary(trace, var_names=["mu_alpha", "b_year_g", "tau", "sigma_u", "rho_bym2", "b_cov"])
'''),
md("""
### What the covariate effects mean
Each coefficient is the change in the log-odds of being zero-dose for a one standard deviation increase in the covariate. A negative value with a 95% interval below zero means LGAs with more of that factor have fewer zero-dose children, other things equal.
"""),
code(r'''
post = trace.posterior.isel(draw=slice(None, None, 8))             # keep every 8th draw: 4,000 draws for summaries
bc = post["b_cov"].values.reshape(-1, len(COVS))
pd.DataFrame({"covariate": COVS, "effect": bc.mean(0), "lower_95": np.percentile(bc, 2.5, 0),
              "upper_95": np.percentile(bc, 97.5, 0), "prob_negative": (bc < 0).mean(0)}).round(3)
'''),
md("""
## 7. Does the model reproduce the surveys? (posterior predictive check)
For each of the 148 survey results, we simulate what a survey of the same size would show if the model were true, including sampling error. About 95% of the actual survey values should fall inside the 95% predictive intervals.
"""),
code(r'''
rng = np.random.default_rng(20261009)
A = post["alpha_s"].values.reshape(-1, len(states)); Bt = post["beta_s"].values.reshape(-1, len(states))
U = post["u_lga"].values.reshape(-1, n); TAU = post["tau"].values.reshape(-1); D = A.shape[0]
ppc = []
for wi, yr in enumerate(years):
    Pst = (M @ (1 / (1 + np.exp(-((A + Bt * t_w[wi])[:, s_idx] + U)))).T).T
    for _, r in obs[obs.wi == wi].iterrows():
        pm_ = np.clip(Pst[:, int(r.si)], 1e-6, 1 - 1e-6); vv = DEFF / (r.n_children_12_23m * pm_ * (1 - pm_))
        sim = 100 / (1 + np.exp(-(np.log(pm_ / (1 - pm_)) + rng.normal(0, 1, D) * np.sqrt(vv + TAU ** 2))))
        lo, hi = np.percentile(sim, [2.5, 97.5])
        ppc.append(dict(state=r.state, year=yr, observed=r.zero_dose_pct, model=pm_.mean() * 100, lo95=lo, hi95=hi))
ppc = pd.DataFrame(ppc); ppc["inside"] = ppc.observed.between(ppc.lo95, ppc.hi95)
print(f"Survey results inside their 95% predictive interval: {ppc.inside.mean():.0%} of {len(ppc)}")
'''),
md(r"""
## 8. LGA estimates for 2026
The 2026 rate for each LGA uses the state's forecast level and trend plus the LGA effect. Because the future can deviate from the trend, a state-level process deviation ($\tau$) is added to every draw. Zero-dose children = rate x LGA cohort. Every quantity is summarised over 4,000 posterior draws.

**Priority probability**: in each draw, LGAs are ranked by zero-dose children; the probability is the share of draws in which the LGA is among the top 155 (20% of LGAs).
"""),
code(r'''
t26 = (2026 - YEAR_CENTER) / ysd
e = rng.normal(0, 1, A.shape) * TAU[:, None]                       # state-level process deviation for 2026
p26 = 1 / (1 + np.exp(-((A + Bt * t26 + e)[:, s_idx] + U)))           # draws x LGAs
N = lga.lga_cohort_12_23m_2024.values
bd = p26 * N
ranks = (-bd).argsort(1).argsort(1) + 1
res = lga[["lga_uid", "zone", "state", "lga", "lga_cohort_12_23m_2024"]].copy()
res["rate_2026"] = p26.mean(0) * 100
res["rate_lo95"], res["rate_hi95"] = np.percentile(p26 * 100, [2.5, 97.5], 0)
res["children_2026"] = bd.mean(0)
res["children_lo95"], res["children_hi95"] = np.percentile(bd, [2.5, 97.5], 0)
res["rank_lo95"], res["rank_hi95"] = np.percentile(ranks, [2.5, 97.5], 0)
res["p_top155"] = (ranks <= 155).mean(0)
res = res.sort_values("children_2026", ascending=False).reset_index(drop=True)
res.insert(0, "national_rank", np.arange(1, n + 1))
nat = bd.sum(1)
print(f"National zero-dose children 2026: {nat.mean():,.0f} (95% credible interval {np.percentile(nat, 2.5):,.0f} to {np.percentile(nat, 97.5):,.0f})")
print("LGAs near-certain to be in the top 155 (probability >= 0.9):", int((res.p_top155 >= 0.9).sum()))
res.head(15).round(2)
'''),
md("## 9. States, zones and burden concentration"),
code(r'''
Pst26 = (M @ p26.T).T * 100
state_tab = pd.DataFrame({"state": states, "zone": st_zone.values, "rate_2026": Pst26.mean(0),
                          "rate_lo95": np.percentile(Pst26, 2.5, 0), "rate_hi95": np.percentile(Pst26, 97.5, 0)})
sb = np.stack([bd[:, s_idx == i].sum(1) for i in range(len(states))], 1)
state_tab["children_2026"] = sb.mean(0); state_tab["children_lo95"], state_tab["children_hi95"] = np.percentile(sb, [2.5, 97.5], 0)
zone_tab = pd.DataFrame([dict(zone=z_, children_2026=bd[:, lga.zone.values == z_].sum(1).mean(),
                              lo95=np.percentile(bd[:, lga.zone.values == z_].sum(1), 2.5),
                              hi95=np.percentile(bd[:, lga.zone.values == z_].sum(1), 97.5)) for z_ in zones])
cum = res.children_2026.cumsum() / res.children_2026.sum() * 100
pareto = {s_: int(np.searchsorted(cum.values, s_) + 1) for s_ in (50, 60, 80)}
print({f"{k}% of zero-dose children": f"{v_} LGAs" for k, v_ in pareto.items()}, f"| top 155 hold {cum.iloc[154]:.1f}%")
display(zone_tab.round(0)); state_tab.sort_values("children_2026", ascending=False).round(1).head(10)
'''),
md("""
## 10. Agreement with independent data
* **NmDHS 2025-26** was not used to fit the model. We compare the state aggregates of the LGA estimates with 100 minus Penta1 coverage from NmDHS (Spearman rank correlation with a bootstrap 95% confidence interval, and mean absolute error).
* **IHME Local Burden of Disease DTP1 (2018)** is an independent LGA map built by another team. It is eight years earlier, so it tests the **pattern**: do LGAs rank in a similar order, nationally and **within each state**?
"""),
code(r'''
from scipy.stats import spearmanr
def spearman_ci(a, b, B=2000, seed=1):
    a, b = np.asarray(a, float), np.asarray(b, float); r = spearmanr(a, b); rg = np.random.default_rng(seed)
    bs = [spearmanr(a[i], b[i]).statistic for i in (rg.integers(0, len(a), len(a)) for _ in range(B))]
    return dict(rho=round(float(r.statistic), 3), p_value=float(r.pvalue), ci95=(round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)), n=len(a))
nm = pd.read_csv(DATA_DIR / "nmdhs_2025_26_penta1_by_state_zone.csv")
vs = state_tab.merge(nm[nm.level == "state"].rename(columns={"area": "state"}), on="state")
print("NmDHS 2025-26 (37 states):", spearman_ci(vs.rate_2026, vs.zd_pct_100_minus_penta1),
      "| mean absolute error:", round((vs.rate_2026 - vs.zd_pct_100_minus_penta1).abs().mean(), 1), "pp")
iv = res.merge(lga[["lga_uid", "ihme_zero_dose_2018_pct"]], on="lga_uid").dropna(subset=["ihme_zero_dose_2018_pct"])
iv["ra"] = iv.groupby("state").rate_2026.rank(pct=True); iv["rb"] = iv.groupby("state").ihme_zero_dose_2018_pct.rank(pct=True)
print("IHME 2018, national LGA ranking:", spearman_ci(iv.rate_2026, iv.ihme_zero_dose_2018_pct))
print("IHME 2018, ranking within states:", spearman_ci(iv.ra, iv.rb))
'''),
md("""
### Why routine coverage alone is not used
Dividing DHIS2 Penta1 doses by the estimated cohort gives "administrative coverage". In most LGAs it exceeds 100%, which is impossible for a true coverage rate; it reflects denominator error and services delivered across LGA boundaries. Subtracting it from 100 would suggest almost no zero-dose children where surveys find many.
"""),
code(r'''
ac = lga.dhis2_admin_penta1_coverage_pct.dropna()
print(f"LGAs with administrative Penta1 coverage above 100%: {(ac > 100).mean():.0%} (median {ac.median():.0f}%)")
'''),
md("## 11. Maps"),
code(r'''
import geopandas as gpd, matplotlib.pyplot as plt
g = gpd.read_file(DATA_DIR / "nga_lgas.geojson").rename(columns={"lga_key": "geo_lga_key"})
gs = gpd.read_file(DATA_DIR / "nga_states.geojson")
g = g.drop(columns=["state", "lga"]).merge(lga[["lga_uid", "state_key", "geo_lga_key"]], on=["state_key", "geo_lga_key"]).merge(res, on="lga_uid")
assert len(g) == 774
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
for a_, col, cm, ttl in [(ax[0], "rate_2026", "YlOrRd", "Zero-dose rate 2026 (%)"), (ax[1], "children_2026", "Reds", "Zero-dose children 2026"),
                         (ax[2], "p_top155", "viridis", "Probability of being in the top 155")]:
    g.plot(ax=a_, column=col, cmap=cm, legend=True, linewidth=0.1, edgecolor="white", legend_kwds={"shrink": 0.6})
    gs.boundary.plot(ax=a_, color="black", linewidth=0.4); a_.set_title(ttl); a_.axis("off")
plt.tight_layout(); plt.show()
'''),
md("## 12. Save outputs"),
code(r'''
res.to_csv(OUT_DIR / "method2_sae_lga_estimates_2026.csv", index=False)
state_tab.to_csv(OUT_DIR / "method2_sae_state_estimates_2026.csv", index=False)
zone_tab.to_csv(OUT_DIR / "method2_sae_zone_estimates_2026.csv", index=False)
ppc.to_csv(OUT_DIR / "method2_sae_posterior_predictive_check.csv", index=False)
summ.to_csv(OUT_DIR / "method2_sae_mcmc_diagnostics.csv")
np.savez_compressed(OUT_DIR / "method2_sae_lga_draws_2026.npz", rate=p26.astype("float32"), lga_uid=lga.lga_uid.values,
                    state=lga.state.values, lga=lga.lga.values, cohort=N, b_cov=bc.astype("float32"), covariates=np.array(COVS),
                    state_rate=Pst26.astype("float32"), states=np.array(states))
json.dump(dict(national_children_2026=float(nat.mean()), national_lo95=float(np.percentile(nat, 2.5)),
               national_hi95=float(np.percentile(nat, 97.5)), pareto=pareto, diagnostics=diag,
               survey_ppc_coverage95=float(ppc.inside.mean())), open(OUT_DIR / "method2_sae_summary.json", "w"), indent=2)
print("Saved:", sorted(p.name for p in OUT_DIR.glob("method2_*")))
'''),
md("""
## Limitations and good practice
* **Ecological assumption.** Covariate effects are learned from differences between states and applied within states.
* **Wide LGA intervals.** No survey measures LGAs directly, so LGA uncertainty is large, especially in the north. Use priority probabilities and intervals, not only point estimates.
* **Denominators** are projections (2024 under-five; NPC 2022 shares). Intervals do not include denominator error.
* **Survey design** is approximated from published tables (design effect 2 assumed).
* **Monte Carlo variation**: results can differ very slightly between runs and computers (well under 1% for totals), because sampling uses random draws and multi-threaded arithmetic.

**Reference:** Riebler A, Sorbye SH, Simpson D, Rue H (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research* 25(4):1145-1165.
"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                  "colab": {"provenance": []}})
nbf.write(nb, "D5_Method2_Bayesian_Small_Area_Estimation.ipynb")
print("ok")
