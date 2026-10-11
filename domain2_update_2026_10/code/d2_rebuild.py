"""Domain 2 rebuild (11.10.2026) with corrected DHIS2 parsing.

The original Domain 2 notebook read DHIS2 counts with pd.to_numeric(errors="coerce"), which silently turns counts written
with a thousands separator ("1,234") into missing values. This script reuses the web app's own functions (data_io.prep_dhis2,
which removes the separator; the same Prophet and LASSO routines) so the figures match the Dropout & Completion page.

Outputs (figures/ and results/):
  D2_F1_dropout_forecasts       national dropout forecasts, three antigen transitions (Prophet, 80% and 95% PI)
  D2_F2_dropout_heatmap         state x year dropout, observed 2021-2025 and Prophet forecast 2026-2027
  D2_F3_dropout_drivers_state   signed LASSO coefficients, 37 states, 2024 (web app: Drivers (LASSO))
  D2_F4_dropout_drivers_lga     signed LASSO coefficients, LGA level 2021-2024 (web app: Drivers - LGA covariates)
  results: parsing_impact.csv, national_dropout_summary.csv, state_year_*.csv, lasso_*.csv, stability_*.csv, summary.json
Run with the web-app environment from the repository root is not needed; paths are absolute.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parents[1] / "nphcda-zerodose-git"
sys.path.insert(0, str(APP))
import logging  # noqa: E402
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

import config as C  # noqa: E402
from data_io import national_monthly, prep_dhis2, state_monthly  # noqa: E402
from models.d2_dropout import (dropout_forecasts, lasso_drivers, lasso_drivers_inference, lga_lasso,  # noqa: E402
                               state_year_with_forecast)

FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
NAVY, INK, GREY, RED, GREEN = "#1F3B57", "#22333F", "#5B6B79", "#C0392B", "#1C7A3D"
PAIR_COL = {"dropout_p1p3": "#E65100", "dropout_p1m1": "#880E4F", "dropout_m1m2": "#4A148C"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlecolor": NAVY, "savefig.dpi": 300})

raw = pd.read_csv(APP / "data" / "sample" / "dhis2_data_all_states.csv", dtype=str)
mds = pd.read_csv(APP / "data" / "sample" / "nigeria_zero_dose_model_dataset.csv")
master = pd.read_csv(APP / "data" / "sample" / "lga_archetype_master.csv")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- parsing impact (old vs corrected)
cc = ["penta_1_count", "penta_3_count", "measles_1_count", "measles_2_count"]
imp = []
for c in cc:
    bad = pd.to_numeric(raw[c], errors="coerce")
    ok = pd.to_numeric(raw[c].str.replace(",", "", regex=False), errors="coerce")
    imp.append({"antigen": c, "rows_lost_pct": round(100 * ((ok.notna()) & (bad.isna())).sum() / ok.notna().sum(), 1),
                "doses_lost_pct": round(100 * (ok.sum() - bad.sum()) / ok.sum(), 1)})
pd.DataFrame(imp).to_csv(RES / "parsing_impact.csv", index=False)

# ---------------------------------------------------------------- corrected series (web-app functions)
d = prep_dhis2(raw)
nat = national_monthly(d)
agg = state_monthly(d)
fc = dropout_forecasts(nat, key="d2_rebuild")

rows = []
for col, s in fc.items():
    obs = pd.Series(s["obs_y"], index=pd.to_datetime(s["obs_x"]))
    fx = pd.to_datetime(s["fore_x"]); fy = np.asarray(s["fore_y"])
    cut = pd.Timestamp(s["cutoff"])
    m12 = ((fx.year - cut.year) * 12 + (fx.month - cut.month)) <= 12
    i12 = np.where(m12)[0][-1]
    yr = obs.groupby(obs.index.year).mean()
    rows.append({"pair": C.DROPOUT_TARGETS[col], "col": col, "last_month": cut.strftime("%b %Y"),
                 "latest_pct": round(float(obs.iloc[-1]), 1), "mean_2021": round(float(yr.get(2021, np.nan)), 1),
                 "mean_2024": round(float(yr.get(2024, np.nan)), 1), "mean_2025": round(float(yr.get(2025, np.nan)), 1),
                 "forecast_12m_pct": round(float(fy[i12]), 1), "forecast_12m_lo95": round(float(s["lo95"][i12]), 1),
                 "forecast_12m_hi95": round(float(s["hi95"][i12]), 1), "forecast_month": fx[i12].strftime("%b %Y")})
NS = pd.DataFrame(rows)
NS.to_csv(RES / "national_dropout_summary.csv", index=False)

# ---------------------------------------------------------------- F1 national forecasts
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
for ax, (col, s) in zip(axes, fc.items()):
    c = PAIR_COL[col]
    ax.scatter(pd.to_datetime(s["obs_x"]), s["obs_y"], s=14, color=c, alpha=0.6, label="Observed (DHIS2)", zorder=3)
    ax.plot(pd.to_datetime(s["hist_x"]), s["hist_y"], color=c, lw=2, label="Fitted")
    fx = pd.to_datetime(s["fore_x"])
    ax.plot(fx, s["fore_y"], "--", color=c, lw=2.2, label="Forecast")
    ax.fill_between(fx, s["lo95"], s["hi95"], color=c, alpha=0.13, label="95% prediction interval")
    ax.fill_between(fx, s["lo80"], s["hi80"], color=c, alpha=0.25, label="80% prediction interval")
    ax.axhline(0, color="#333", ls=":", lw=1); ax.axvline(pd.Timestamp(s["cutoff"]), color="#888", ls=":", lw=1)
    ax.set_title(f"{s['label']} dropout"); ax.set_ylabel("Dropout (%)"); ax.grid(alpha=0.15, ls="--")
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("National dropout rates: observed DHIS2 monthly series and Prophet forecast", fontsize=13, fontweight="bold", color=NAVY)
fig.text(0.01, -0.03, "Model estimate. DHIS2 2021-2025, counts parsed with thousands separators; negative dropout means the later dose "
         "was recorded at or above the earlier dose in that month.", fontsize=8.5, style="italic", color=GREY)
fig.tight_layout()
save(fig, "D2_F1_dropout_forecasts")

# ---------------------------------------------------------------- F2 state x year heatmap (observed + forecast)
pivs = {}
for metric in ("dropout_p1p3", "dropout_m1m2"):
    p = state_year_with_forecast(agg, metric, key="d2_rebuild_" + metric)
    p.columns = [int(c) for c in p.columns]
    pivs[metric] = p
    p.to_csv(RES / f"state_year_{metric}.csv")
cov = (agg.assign(has=agg[["penta_1_count"]].notna()).groupby(["state", "year"]).size().unstack())
fig, axes = plt.subplots(1, 2, figsize=(19, 13.5))
norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
for ax, metric in zip(axes, ("dropout_p1p3", "dropout_m1m2")):
    p = pivs[metric]
    ax.set_facecolor("#E5E7EB")
    im = ax.imshow(np.ma.masked_invalid(p.values), cmap="RdYlGn_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(p.columns)), [str(c) for c in p.columns])
    ax.set_yticks(range(len(p.index)), p.index, fontsize=9)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.4,
                        color="white" if abs(v) > 22 else INK, fontweight="bold" if p.columns[j] > 2025 else "normal")
            else:
                ax.text(j, i, "no data", ha="center", va="center", fontsize=7, color=GREY, style="italic")
    nobs = sum(c <= 2025 for c in p.columns)
    ax.axvline(nobs - 0.5, color="black", lw=2)
    ax.text((nobs - 1) / 2, -0.9, "OBSERVED 2021-2025", ha="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(nobs + 0.5, -0.9, "FORECAST 2026-2027", ha="center", fontsize=9, fontweight="bold", color=NAVY)
    ax.set_title(f"{C.DROPOUT_TARGETS[metric]} dropout by state and year (%)", pad=22)
    for s in ax.spines.values():
        s.set_visible(False)
cb = fig.colorbar(im, ax=axes, shrink=0.6, extend="both", pad=0.02)
cb.set_label("Dropout (%); colours capped at +/-30")
fig.text(0.01, 0.04, "Observed: mean of monthly state dropout from DHIS2 (counts parsed with thousands separators). Grey = no DHIS2 records for the state that year. Forecast: calendar-year "
         "mean of monthly Prophet forecasts per state. Negative values mean the later dose was recorded at or above the earlier dose.",
         fontsize=8.5, style="italic", color=GREY)
save(fig, "D2_F2_dropout_heatmap")

# ---------------------------------------------------------------- F3 state LASSO drivers (signed), F4 LGA LASSO drivers
drv = lasso_drivers(agg, mds)
inf = lasso_drivers_inference(agg, mds, key="d2_rebuild", n_boot=200, top_k=4)
lg = lga_lasso(raw, master, key="d2_rebuild")
for t, s in drv.items():
    s.rename("coef").to_csv(RES / f"lasso_state_{t}.csv")
for t, r in lg.items():
    r["coef"].rename("coef").to_csv(RES / f"lasso_lga_{t}.csv")
for lab, df in inf.items():
    df.to_csv(RES / f"stability_{lab.replace(' ', '_')}.csv", index=False)


def lasso_panel(fig_title, series_by_target, note, name, sub=None):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    for ax, t in zip(axes, C.DROPOUT_TARGETS):
        s = series_by_target.get(t, pd.Series(dtype=float))
        s = s[s.abs() > 1e-9]
        top = s.reindex(s.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
        if len(top) == 0:
            ax.text(0.5, 0.5, "No covariate retained\nby the cross-validated LASSO", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color=GREY)
            ax.set_yticks([]); ax.set_xticks([])
        else:
            ax.barh([C.feature_label(f) for f in top.index], top.values,
                    color=[RED if v > 0 else GREEN for v in top.values], height=0.62)
            m = float(top.abs().max())
            ax.set_xlim(-1.5 * m, 1.5 * m); ax.axvline(0, color="#64748B", lw=1)
            for i, v in enumerate(top.values):
                ax.text(v + (0.03 * m if v >= 0 else -0.03 * m), i, f"{v:+.2f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=8.5)
            ax.tick_params(axis="y", labelsize=9)
        ax.set_title(f"{C.DROPOUT_TARGETS[t]} dropout" + (f"\n{sub[t]}" if sub else ""), fontsize=10.5)
        ax.set_xlabel("Standardized LASSO coefficient")
    fig.suptitle(fig_title, fontsize=13, fontweight="bold", color=NAVY)
    fig.text(0.01, -0.04, note, fontsize=8.5, style="italic", color=GREY)
    fig.tight_layout()
    save(fig, name)


lasso_panel("Drivers of dropout across states (LASSO, 2024)", drv,
            "Red = associated with higher dropout; green = lower dropout. 37 states; 2024 state dropout from DHIS2 regressed on NDHS "
            "state equity covariates (standardized, 5-fold cross-validated LASSO). Ecological associations, not causal effects.",
            "D2_F3_dropout_drivers_state")
lasso_panel("Drivers of dropout across LGAs (LASSO on 15 LGA covariates, 2021-2024)", {t: r["coef"] for t, r in lg.items()},
            "Red = associated with higher dropout; green = lower dropout. LGA dropout from DHIS2 totals 2021-2024 regressed on 15 "
            "standardized LGA covariates (IHME, DHS, Meta, Weiss, ACLED). Ecological associations, not causal effects.",
            "D2_F4_dropout_drivers_lga",
            sub={t: f"{r['n']} LGAs; R-squared {r['r2']:.2f}" for t, r in lg.items()})

summary = {"parsing_impact": imp, "national": NS.to_dict(orient="records"),
           "state_lasso": {t: s.round(3).to_dict() for t, s in drv.items()},
           "lga_lasso": {t: {"coef": r["coef"].round(3).to_dict(), "n": r["n"], "r2": round(r["r2"], 3),
                             "median_dropout": round(r["median_dropout"], 1)} for t, r in lg.items()},
           "stability": {k: v.to_dict(orient="records") for k, v in inf.items()},
           "state_year": {m: {"neg_cells_obs": int((p[[c for c in p.columns if c <= 2025]] < 0).sum().sum()),
                              "states_above_10_2025": int((p[2025] > 10).sum()) if 2025 in p.columns else None,
                              "top5_2025": p[2025].sort_values(ascending=False).head(5).round(1).to_dict() if 2025 in p.columns else {},
                              "median_2025": round(float(p[2025].median()), 1) if 2025 in p.columns else None,
                              "median_2027": round(float(p[2027].median()), 1) if 2027 in p.columns else None}
                          for m, p in pivs.items()}}
(RES / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=1, default=str)[:6000])
