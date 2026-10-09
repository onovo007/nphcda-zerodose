"""Deck-specific figure variants (layouts sized for slide panels). Reuses figures.py data and styling."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import figures as F  # noqa: E402

OUT = F.ROOT / "figures" / "deck"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    g, gs = F.load_geo()
    hs = pd.read_csv(F.FIG / "data" / "M1_01_lga_rate_and_hotspots.csv")
    g = g.merge(hs[["state", "lga_clean", "gi"]], on=["state", "lga_clean"], how="left")
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for k, c in {**F.HOT, "Not estimated": "#BBBBBB"}.items():
        sub = g[g.gi == k]
        if len(sub):
            sub.plot(ax=ax, color=c, edgecolor="white", linewidth=0.08)
    F.basemap(ax, g, gs)
    cnt = g.gi.value_counts()
    ax.legend(handles=[Patch(color=c, label=f"{k} ({cnt.get(k, 0)})") for k, c in {**F.HOT, "Not estimated": "#BBBBBB"}.items()
                       if cnt.get(k, 0)], fontsize=7.5, frameon=False, loc="lower right", title="LGA clusters", title_fontsize=8)
    ax.set_title("Zero-dose hotspot clusters, 2026 (Method 1)", loc="left", fontsize=11)
    fig.savefig(OUT / "D_M1_lga_hotspots.png", bbox_inches="tight", facecolor="white", pad_inches=0.05, dpi=300)
    plt.close(fig)
    print("ok", cnt.to_dict())


def state_figures():
    """Report figures 9-10 from the canonical Method 1 notebook outputs (state forecasts 2026-2028) and NDHS observations."""
    import numpy as np
    f = pd.read_csv(F.ROOT / "notebooks" / "outputs_Method1_Bayesian_Hierarchical_Model" / "method1_state_forecasts_2026_2028.csv")
    obs = pd.read_parquet(F.C.D["bayes"] / "D_corrected_balanced_trend" / "state_results_original_format.parquet")[
        ["state", "zd_obs_2008", "zd_obs_2013", "zd_obs_2018", "zd_obs_2024"]]
    d = f.merge(obs, on="state", how="left")
    assert d.zd_obs_2024.notna().all() and len(d) == 37
    zones = ["North West", "North East", "North Central", "South West", "South South", "South East"]
    ZC = dict(zip(zones, ["#7F1D1D", "#D55E00", "#E6A817", "#56B4E9", "#0072B2", "#1C7A3D"]))
    d["zo"] = d.zone.map({z: i for i, z in enumerate(zones)})
    d = d.sort_values(["zo", "rate_2026"], ascending=[True, False]).reset_index(drop=True)
    # Figure 9: small multiples
    fig, axs = plt.subplots(5, 8, figsize=(16, 10.5), sharex=True, sharey=True)
    for ax, r in zip(axs.flat, d.itertuples()):
        yo = [r.zd_obs_2008, r.zd_obs_2013, r.zd_obs_2018, r.zd_obs_2024]
        ax.plot([2008, 2013, 2018, 2024], yo, "o-", color="#5B6B79", ms=3, lw=1)
        yrs = [2024, 2026, 2027, 2028]
        m = [r.zd_obs_2024, r.rate_2026, r.rate_2027, r.rate_2028]
        lo = [r.zd_obs_2024, r.rate_2026_lo95, r.rate_2027_lo95, r.rate_2028_lo95]
        hi = [r.zd_obs_2024, r.rate_2026_hi95, r.rate_2027_hi95, r.rate_2028_hi95]
        ax.fill_between(yrs, lo, hi, color=ZC[r.zone], alpha=0.18, lw=0)
        ax.plot(yrs, m, "-", color=ZC[r.zone], lw=1.8)
        ax.set_title(r.state, fontsize=9, color=ZC[r.zone], loc="left")
        ax.set_ylim(0, 100); ax.set_xlim(2006, 2029); ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#EEF2F6")
    for ax in list(axs.flat)[len(d):]:
        ax.set_axis_off()
    fig.legend(handles=[Patch(color=ZC[z], label=z) for z in zones] +
               [plt.Line2D([], [], color="#5B6B79", marker="o", ms=3, label="NDHS observed")],
               loc="lower right", bbox_to_anchor=(0.98, 0.04), fontsize=9, frameon=False, title="Zone; line = 2026-2028 forecast with 95% interval")
    fig.supylabel("Zero-dose, % of children 12-23 months", fontsize=10)
    fig.suptitle("State zero-dose: NDHS 2008-2024 and Method 1 forecasts 2026-2028", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=F.NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "R_09_state_trajectories.png", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    # Figure 10: 2026 rate with interval
    e = d.sort_values("rate_2026").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    y = np.arange(len(e))
    ax.hlines(y, e.rate_2026_lo95, e.rate_2026_hi95, color=[ZC[z] for z in e.zone], lw=2.2, alpha=0.55)
    ax.scatter(e.rate_2026, y, color=[ZC[z] for z in e.zone], s=28, zorder=3)
    ax.scatter(e.ndhs_2024, y, marker="|", color="#27343D", s=60, zorder=4, label="NDHS 2024 observed")
    ax.set_yticks(y); ax.set_yticklabels(e.state, fontsize=8.5)
    ax.set_xlabel("Zero-dose rate 2026, % (posterior mean and 95% credible interval)")
    ax.legend(handles=[Patch(color=ZC[z], label=z) for z in zones] +
              [plt.Line2D([], [], color="#27343D", marker="|", ls="", ms=9, label="NDHS 2024 observed")], fontsize=8.5, frameon=False,
              loc="lower right")
    ax.set_title("Method 1 state forecasts, 2026", loc="left")
    ax.grid(axis="x", color="#EEF2F6")
    fig.savefig(OUT / "R_10_state_2026_intervals.png", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print("state figures ok")


if __name__ == "__main__":
    main()
    state_figures()
