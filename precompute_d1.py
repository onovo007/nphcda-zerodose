"""One-time: per-LGA Prophet early-warning forecasts on the BUNDLED sample data, saved so the
deployed app serves them instead of fitting about 3,100 Prophet models live (tens of minutes on a
small shared container). Uploaded data still uses the live screen. Re-run only if the sample data or
the method changes:  python precompute_d1.py [--jobs 20]

Method (same as the national forecasts and the Domain 1 notebook):
  * Prophet, yearly + semi-annual seasonality (period 182.5, Fourier order 3), interval_width 0.95,
    changepoint_prior_scale 0.05, seasonality_prior_scale 10, 18-month horizon, monthly frequency;
  * each LGA series expressed as a percent of its own 2024 mean monthly doses; early-warning flag when
    the central forecast falls below 80% in months 6-12 after the last observed month;
  * LGA filters: at least 18 monthly rows, non-zero total, positive 2024 baseline.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config as C  # noqa: E402
import data_io as io  # noqa: E402
import names as N  # noqa: E402
from models.d1_forecast import _fp_d1, _PRECOMP  # noqa: E402

HORIZON = 18


def _fit(ts: pd.DataFrame) -> pd.DataFrame:
    from prophet import Prophet
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                interval_width=0.95, changepoint_prior_scale=0.05, seasonality_prior_scale=10)
    m.add_seasonality(name="semi_annual", period=182.5, fourier_order=3)
    m.fit(ts)
    return m.predict(m.make_future_dataframe(periods=HORIZON, freq="MS"))


def fit_lga(key, sub: pd.DataFrame):
    zone, state, lga = key
    rows, monthly, skipped = [], [], []
    lga_name = N.clean_lga_name(lga)
    if len(sub) < 18:
        return rows, monthly, [dict(State=state, LGA=lga_name, Antigen=a,
                                    Reason="fewer than 18 monthly rows") for a in C.ANTIGEN_TS]
    for antigen, col in C.ANTIGEN_TS.items():
        ts = sub[["ds", col]].rename(columns={col: "y"}).dropna()
        if ts["y"].sum() == 0 or len(ts) < 18:
            skipped.append(dict(State=state, LGA=lga_name, Antigen=antigen,
                                Reason="no doses or fewer than 18 months"))
            continue
        base = ts.loc[ts["ds"].dt.year == 2024, "y"].mean()
        if not np.isfinite(base) or base <= 0:
            skipped.append(dict(State=state, LGA=lga_name, Antigen=antigen,
                                Reason="no 2024 baseline"))
            continue
        fc = _fit(ts)
        cutoff = ts["ds"].max()
        fore = fc[fc["ds"] > cutoff].copy()
        fore["pct"] = fore["yhat"] / base * 100
        fore["ahead"] = (fore["ds"].dt.year - cutoff.year) * 12 + (fore["ds"].dt.month - cutoff.month)
        win = fore[(fore["ahead"] >= C.AT_RISK_WINDOW_MONTHS[0]) & (fore["ahead"] <= C.AT_RISK_WINDOW_MONTHS[1])]
        if win.empty:
            skipped.append(dict(State=state, LGA=lga_name, Antigen=antigen,
                                Reason="no forecast months in window"))
            continue
        below = fore[fore["pct"] < C.THRESHOLD_PCT]
        rows.append({"Zone": zone, "State": state, "LGA": lga_name, "Antigen": antigen,
                     "2024 baseline (doses/month)": round(float(base), 1),
                     "Lowest forecast, months 6-12 (% of 2024)": round(float(win["pct"].min()), 1),
                     "Month of lowest forecast": win.loc[win["pct"].idxmin(), "ds"].strftime("%b %Y"),
                     "Early-warning flag": bool((win["pct"] < C.THRESHOLD_PCT).any()),
                     "First month below 80%": below["ds"].min().strftime("%b %Y") if len(below) else "-",
                     "Months observed": int(len(ts)), "Last observed": cutoff.strftime("%b %Y")})
        fm = fore[fore["ds"].dt.year.isin([2026, 2027])]
        monthly.append(pd.DataFrame({
            "zone": zone, "state": state, "lga": lga_name, "antigen": antigen,
            "month": fm["ds"].dt.strftime("%Y-%m"), "year": fm["ds"].dt.year.astype(int),
            "forecast_doses": fm["yhat"].clip(lower=0).round(0),
            "lower95_doses": fm["yhat_lower"].clip(lower=0).round(0),
            "upper95_doses": fm["yhat_upper"].clip(lower=0).round(0),
            "pct_of_2024_baseline": fm["pct"].round(1)}))
    return rows, monthly, skipped


def severity(pct: pd.Series, flag: pd.Series) -> pd.Series:
    band = pd.cut(pct, [-np.inf, 0, 40, 70, 80],
                  labels=["Unstable forecast (<0%)", "Critical (0-40%)", "Severe (40-70%)",
                          "Warning (70-80%)"]).astype(str)
    return band.where(flag, "No flag")


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    _PRECOMP.mkdir(parents=True, exist_ok=True)
    raw = io.load_sample()["dhis2"]
    d = io.prep_dhis2(raw)
    cols = list(C.ANTIGEN_TS.values())
    agg = (d.groupby(["zone", "state", "lga", "ds"])[cols].sum().reset_index()
           .sort_values(["state", "lga", "ds"]))
    groups = list(agg.groupby(["zone", "state", "lga"], sort=False))
    print(f"Fitting Prophet for {len(groups)} LGAs x {len(cols)} antigens ...")
    res = Parallel(n_jobs=a.jobs)(delayed(fit_lga)(k, g) for k, g in groups)
    summ = pd.DataFrame([r for o, _, _ in res for r in o])
    monthly = pd.concat([m for _, ms, _ in res for m in ms], ignore_index=True)
    skipped = pd.DataFrame([s for _, _, sk in res for s in sk])
    summ["Severity band"] = severity(summ["Lowest forecast, months 6-12 (% of 2024)"],
                                     summ["Early-warning flag"])
    summ.to_parquet(_PRECOMP / "d1_lga_prophet.parquet")
    monthly.to_csv(_PRECOMP / "d1_lga_prophet_monthly.csv", index=False)  # text: Space binary limits
    skipped.to_parquet(_PRECOMP / "d1_lga_not_forecast.parquet")
    flags = summ[summ["Early-warning flag"]]
    meta = {"fp": _fp_d1(raw), "forecasts": int(len(summ)), "lgas": int(summ["LGA"].nunique()),
            "lgas_key": int(summ.groupby(["State", "LGA"]).ngroups),
            "alerts": int(len(flags)), "lgas_with_alert": int(flags.groupby(["State", "LGA"]).ngroups),
            "alerts_by_antigen": flags["Antigen"].value_counts().to_dict(),
            "severity": flags["Severity band"].value_counts().to_dict(),
            "last_observed": summ["Last observed"].mode().iloc[0]}
    (_PRECOMP / "d1_lga_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"Done in {time.time() - t0:.0f}s -> {_PRECOMP}")


if __name__ == "__main__":
    main()
