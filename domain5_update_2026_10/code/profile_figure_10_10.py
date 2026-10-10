"""Profile map and burden-share figure with the evidence-based profile labels (replaces A_01 in the 10.10.2026 deck and report).
Writes figures/P_01_profile_map_burden.{png,pdf}. Reads results and geometry only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
DSA = ROOT / "Updated Datasets for Domain 5" / "LGA_Archetypes"
PLAB = {1: "Deprived northern rural", 2: "Low maternal care, high conflict exposure", 3: "Geographically isolated",
        4: "Near-average", 5: "Relatively advantaged"}
PCOL = {1: "#8B1A1A", 2: "#D6604D", 3: "#2C7FB8", 4: "#F2C14E", 5: "#1C7A3D"}
M1C, M2C, INK = "#0072B2", "#D55E00", "#1A1A1A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

lg = pd.read_csv(RES / "lga_both_methods_774.csv")
lg = lg.loc[:, ~lg.columns.duplicated()]
arch = pd.read_csv(RES / "archetype_both_methods.csv").sort_values("archetype").reset_index(drop=True)
g = gpd.read_file(DSA / "nga_lgas.geojson")
gs = gpd.read_file(DSA / "nga_states.geojson")
keys = pd.read_csv(DSA / "lga_map_keys_774.csv")[["lga_uid", "state_key", "geo_lga_key"]]
gk = g.merge(keys.rename(columns={"geo_lga_key": "lga_key"}), on=["state_key", "lga_key"]).merge(lg[["lga_uid", "archetype"]], on="lga_uid")

fig = plt.figure(figsize=(14, 7))
ax0 = fig.add_axes([0.0, 0.2, 0.47, 0.75])
gk.plot(ax=ax0, color=gk.archetype.map(PCOL), edgecolor="white", linewidth=0.08)
gs.boundary.plot(ax=ax0, color="#333", linewidth=0.45)
ax0.axis("off")
ax0.legend(handles=[Patch(color=PCOL[k], label=f"{k}  {PLAB[k]} ({int((lg.archetype == k).sum())})") for k in range(1, 6)],
           loc="upper left", fontsize=9, frameon=False, bbox_to_anchor=(0.0, 0.02))
ax0.set_title("A. Five contextual profiles, 774 LGAs", loc="left", fontsize=11, fontweight="bold")
ax1 = fig.add_axes([0.55, 0.25, 0.43, 0.65])
x = np.arange(5); w = 0.27
ax1.bar(x - w, arch.cohort_share, w, color="#C9D3DC", label="Share of children 12-23 months")
ax1.bar(x, arch.m1_share, w, color=M1C, label="Share of zero-dose, Method 1")
ax1.bar(x + w, arch.m2_share, w, color=M2C, label="Share of zero-dose, Method 2")
for xi, (a1, a2) in enumerate(zip(arch.m1_children, arch.m2_children)):
    ax1.text(xi, max(arch.m1_share[xi], arch.m2_share[xi], arch.cohort_share[xi]) + 1.2, f"{a1 / 1e3:,.0f}k | {a2 / 1e3:,.0f}k",
             ha="center", fontsize=8.5, color=INK)
ax1.set_xticks(x); ax1.set_xticklabels(["Deprived\nnorthern rural", "Low maternal care,\nhigh conflict", "Geographically\nisolated",
                                         "Near-average", "Relatively\nadvantaged"], fontsize=8.5)
ax1.set_ylabel("%"); ax1.legend(frameon=False, fontsize=8.5)
ax1.set_title("B. Burden is concentrated beyond population share", loc="left", fontsize=11, fontweight="bold")
fig.text(0.01, 0.02, "Model estimate, 2026. Ward hierarchical clustering of 15 contextual indicators. Labels above bars: Method 1 | Method 2 zero-dose children.",
         fontsize=8, style="italic", color="#555")
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"P_01_profile_map_burden.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("saved")
