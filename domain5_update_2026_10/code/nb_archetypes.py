"""Build notebook: Domain 5 LGA structural archetypes (unsupervised clustering), all 774 LGAs."""
import nbformat as nbf

from nb_common import code, md, setup_cell

REQ = ["lga_archetype_covariates_774.csv", "archetype_covariate_sources.csv", "lga_zero_dose_estimates_2026_both_methods.csv",
       "lga_map_keys_774.csv", "nga_lgas.geojson", "nga_states.geojson"]
PIP = ["geopandas", "scikit-learn", "openpyxl"]

cells = [
md("""
# Domain 5: Community archetypes for all 774 LGAs (unsupervised machine learning)

**National Primary Health Care Development Agency (NPHCDA) - Predictive Modelling of Zero-Dose Children.**
Prepared by the CIDRE and Quantium Insights LLC consortium, with UNICEF and Gavi.

**Why archetypes?** Two LGAs can have the same zero-dose rate for very different reasons: one may be remote, another conflict-affected, another an urban area with informal settlements. Each needs a different response. Archetypes group LGAs that share a **profile of structural barriers**, so that an intervention package can be matched to each group.

**How:** agglomerative (Ward) hierarchical clustering of 15 social, health-service, nutrition, geographic and security covariates for all 774 LGAs. Ward clustering starts with every LGA as its own group and repeatedly merges the two groups whose merger increases within-group variation the least, until five groups remain. Modelled zero-dose estimates are **not** used to form the groups; they are attached afterwards to describe each group.

**How to run:** Runtime > Run all and upload the files from the folder `LGA_Archetypes` of *Updated Datasets for Domain 5*.
"""),
setup_cell(REQ, PIP),
md("## 1. The 15 covariates"),
code(r'''
import numpy as np, pandas as pd
m = pd.read_csv(DATA_DIR / "lga_archetype_covariates_774.csv")
src = pd.read_csv(DATA_DIR / "archetype_covariate_sources.csv")
est = pd.read_csv(DATA_DIR / "lga_zero_dose_estimates_2026_both_methods.csv")
covs = list(src.covariate)
assert len(m) == 774 and m.lga_uid.is_unique and len(covs) == 15
print("LGAs:", len(m), "| covariates:", len(covs), "| LGAs with any missing covariate:", int(m[covs].isna().any(axis=1).sum()))
src[["covariate", "measure", "year", "higher_is"]]
'''),
md("""
## 2. Preparing the covariates
* Missing values (a few LGAs whose names did not match a source) are filled with the national median.
* Conflict events and fatalities are very skewed, so they are log-transformed, log(1 + x).
* All covariates are **standardised** (mean 0, standard deviation 1) so that each contributes equally to the distance between LGAs.
"""),
code(r'''
from sklearn.preprocessing import StandardScaler
X = m[covs].copy().fillna(m[covs].median())
for c in ("conflict_events", "conflict_fatalities"):
    X[c] = np.log1p(X[c])
Xz = StandardScaler().fit_transform(X)
print("Standardised matrix:", Xz.shape)
'''),
md("""
## 3. Choosing the number of groups and clustering
Cluster validity indices are computed for 2 to 9 groups. The **silhouette** (higher is better, -1 to 1) measures how much closer each LGA is to its own group than to the next; the **Calinski-Harabasz** index (higher is better) compares between-group to within-group spread. Five groups were chosen for programme use (one package per group), balancing these indices against interpretability.
"""),
code(r'''
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
k_tab = []
for k in range(2, 10):
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xz)
    k_tab.append(dict(k=k, silhouette=silhouette_score(Xz, lab), calinski_harabasz=calinski_harabasz_score(Xz, lab)))
display(pd.DataFrame(k_tab).round(3))

labels = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(Xz)
# Order groups by average deprivation (mean z-score, signed so that higher = worse): archetype 1 = most deprived
sign = np.array([1.0 if h == "worse" else -1.0 for h in src.higher_is])
dep = pd.Series((Xz * sign).mean(1)).groupby(labels).mean().sort_values(ascending=False)
rank = {c: i + 1 for i, c in enumerate(dep.index)}
m["archetype_reproduced"] = [rank[c] for c in labels]
print("Agreement with the published archetypes (adjusted Rand index, 1 = identical):",
      round(adjusted_rand_score(m.archetype, m.archetype_reproduced), 3))
NAMES = {1: "Remote Rural / Hard-to-Reach", 2: "Conflict-Affected / Nomadic", 3: "Riverine / Geographically Isolated",
         4: "Peri-urban / Migrant Dense", 5: "Urban Slums (better-off core)"}
m["archetype_name"] = m.archetype_reproduced.map(NAMES)
m.archetype_name.value_counts()
'''),
md("## 4. What defines each archetype (profiles)"),
code(r'''
prof = m.groupby("archetype_reproduced")[covs].mean()
zprof = pd.DataFrame(Xz, columns=covs).groupby(m.archetype_reproduced.values).mean()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 4.5))
im = ax.imshow(zprof.values, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
ax.set_yticks(range(5)); ax.set_yticklabels([f"A{k} {NAMES[k]}" for k in zprof.index])
ax.set_xticks(range(15)); ax.set_xticklabels(covs, rotation=45, ha="right")
for i in range(5):
    for j in range(15):
        ax.text(j, i, f"{zprof.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, label="mean standardised value"); ax.set_title("Archetype profiles (standardised means)"); plt.tight_layout(); plt.show()
prof.round(2).T
'''),
md("""
## 5. Robustness
* **Vaccination-related covariate.** One covariate (`dpt1_3_dropout`, IHME DPT1-to-DPT3 dropout, 2016) is a vaccination measure. Re-clustering without it shows how much the groups depend on it.
* **Alternative algorithm.** K-means with five groups on the same data.
"""),
code(r'''
from sklearn.cluster import KMeans
c14 = [c for c in covs if c != "dpt1_3_dropout"]
X14 = StandardScaler().fit_transform(X[c14])
lab14 = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(X14)
labkm = KMeans(n_clusters=5, n_init=50, random_state=1).fit_predict(Xz)
print("ARI without the vaccination covariate:", round(adjusted_rand_score(m.archetype_reproduced, lab14), 3))
print("ARI Ward vs K-means (k = 5):", round(adjusted_rand_score(m.archetype_reproduced, labkm), 3))
'''),
md("## 6. Zero-dose burden by archetype (both methods)"),
code(r'''
d = m.merge(est, on="lga_uid", how="left", suffixes=("", "_e"))
tab = d.groupby(["archetype_reproduced", "archetype_name"]).agg(
    lgas=("lga_uid", "size"), children_12_23m=("cohort_12_23m", "sum"),
    method1_zero_dose=("method1_zero_dose_children", "sum"), method2_zero_dose=("method2_zero_dose_children", "sum"),
    method1_mean_rate=("method1_zero_dose_rate_pct", "mean"), method2_mean_rate=("method2_zero_dose_rate_pct", "mean"),
    anc4plus=("anc4plus", "mean"), facility_delivery=("delivery_hf", "mean"), travel_time=("travel_time_hc", "mean"),
    women_education_years=("edu_mean_years_women_15_49", "mean"), conflict_fatalities=("conflict_fatalities", "mean")).reset_index()
for c in ("children_12_23m", "method1_zero_dose", "method2_zero_dose"):
    tab[c + "_share_pct"] = tab[c] / tab[c].sum() * 100
v = d.dropna(subset=["anc4plus", "method1_zero_dose_rate_pct"])
print("Correlation of ANC4+ with the Method 1 LGA zero-dose rate:", round(np.corrcoef(v.anc4plus, v.method1_zero_dose_rate_pct)[0, 1], 2), f"(n = {len(v)})")
tab.round(1)
'''),
md("## 7. Map"),
code(r'''
import geopandas as gpd
from matplotlib.patches import Patch
keys = pd.read_csv(DATA_DIR / "lga_map_keys_774.csv")
g = gpd.read_file(DATA_DIR / "nga_lgas.geojson").rename(columns={"lga_key": "geo_lga_key"}).drop(columns=["state", "lga"])
g = g.merge(keys[["lga_uid", "state_key", "geo_lga_key"]], on=["state_key", "geo_lga_key"]).merge(m[["lga_uid", "archetype_reproduced"]], on="lga_uid")
gs = gpd.read_file(DATA_DIR / "nga_states.geojson")
COL = {1: "#7F1D1D", 2: "#D55E00", 3: "#56B4E9", 4: "#E6A817", 5: "#1C7A3D"}
fig, ax = plt.subplots(figsize=(9, 8))
g.plot(ax=ax, color=g.archetype_reproduced.map(COL), edgecolor="white", linewidth=0.1)
gs.boundary.plot(ax=ax, color="black", linewidth=0.4); ax.axis("off")
ax.legend(handles=[Patch(color=COL[k], label=f"{k} {NAMES[k]} ({(m.archetype_reproduced == k).sum()})") for k in range(1, 6)], loc="lower left", frameon=False)
ax.set_title("Structural archetypes, 774 LGAs"); plt.show()
'''),
md("""
## 8. Matched intervention packages
| Archetype | Defining barriers | Package |
|---|---|---|
| 1 Remote Rural / Hard-to-Reach | lowest maternal education and facility delivery, highest child undernutrition | Reaching-Every-Community microplanning, BHCPF outreach and community health workers; co-deliver with nutrition and antenatal care |
| 2 Conflict-Affected / Nomadic | highest conflict fatalities, weak antenatal and delivery care | Security-integrated microplanning, negotiated access, mobile and transit-point teams |
| 3 Riverine / Geographically Isolated | extreme travel time, low improved water | Boat-based outreach, community health worker networks, multi-antigen bundling per visit |
| 4 Peri-urban / Migrant Dense | moderate on all fronts | Ward enumeration, migrant-sensitive scheduling, Periodic Intensification of Routine Immunization |
| 5 Urban Slums (better-off core) | highest wealth and education; residual informal-settlement gaps | Targeted social mobilisation, fixed-plus-outreach hybrid, private-sector last-mile reach |
"""),
code(r'''
within = m.groupby("state").archetype_reproduced.nunique()
print("States containing 2 or more archetypes:", int((within >= 2).sum()), "| 3 or more:", int((within >= 3).sum()), "of", len(within))
m[["lga_uid", "zone", "state", "lga", "archetype_reproduced", "archetype_name"] + covs].to_csv(OUT_DIR / "lga_archetypes_774.csv", index=False)
tab.to_csv(OUT_DIR / "archetype_profiles_and_burden.csv", index=False)
print("Saved:", sorted(p.name for p in OUT_DIR.glob("*.csv")))
'''),
md("""
## Limitations
* The covariates are modelled surfaces (2016-2024) summarised to LGAs, not direct measurements.
* One covariate is vaccination-related (DPT dropout); results without it are shown above.
* Cluster boundaries are statistical; the archetype names are interpretations of each group's profile. A given LGA may share features of more than one archetype.
* Matched packages are programme recommendations based on each profile; their effectiveness has not been evaluated here.
"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                  "colab": {"provenance": []}})
nbf.write(nb, "D5_LGA_Archetypes_Unsupervised_Clustering.ipynb")
print("ok")
