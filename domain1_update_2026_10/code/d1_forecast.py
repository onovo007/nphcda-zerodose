"""Domain 1: antigen early-warning forecasts (Prophet), national and LGA level.

Same methodology as the Domain 1 notebook (D1_D2__domains_1_2_final.ipynb, Section 5):
  * Prophet, yearly seasonality + semi-annual seasonality (period 182.5, Fourier order 3), interval_width 0.95,
    changepoint_prior_scale 0.05, seasonality_prior_scale 10, 18-month horizon, monthly frequency;
  * each series expressed as % of its own 2024 mean monthly doses; early-warning flag when the central forecast
    falls below 80% in months 6-12 after the last observed month (Dec 2025);
  * LGA filters: at least 18 monthly rows, non-zero total, positive 2024 baseline.
The only change is how dose counts are read:
  --parse original   pd.to_numeric(errors="coerce")  (counts printed with a thousands separator become missing)
  --parse corrected  thousands="," (all counts read)
Usage: python d1_forecast.py --parse corrected
"""
import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "All I Need" / "03_Datasets" / "datasets"
ANTIGENS = {"BCG": "bcg_count", "Penta1": "penta_1_count", "Penta3": "penta_3_count", "Measles1": "measles_1_count"}
COUNT_COLS = ["bcg_count", "penta_1_count", "pent_2_count", "penta_3_count", "measles_1_count", "measles_2_count"]
FORECAST_MONTHS, THRESHOLD, WINDOW = 18, 80.0, (6, 12)


def load(parse):
    if parse == "original":
        d = pd.read_csv(DATA / "dhis2_data_all_states.csv")
        for c in COUNT_COLS:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    else:
        d = pd.read_csv(DATA / "dhis2_data_all_states.csv", thousands=",", low_memory=False)
        for c in COUNT_COLS:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["ds"] = pd.to_datetime(d["period"].astype(str).str.strip(), format="%b-%y", errors="coerce")
    return d


def prophet_fit(df, periods=FORECAST_MONTHS):
    from prophet import Prophet
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95,
                changepoint_prior_scale=0.05, seasonality_prior_scale=10)
    m.add_seasonality(name="semi_annual", period=182.5, fourier_order=3)
    m.fit(df)
    return m.predict(m.make_future_dataframe(periods=periods, freq="MS"))


def national(d):
    agg = d.groupby(["state", "zone", "ds"])[COUNT_COLS].sum().reset_index()
    nat = agg.groupby("ds")[list(ANTIGENS.values())].sum().reset_index()
    rows, fcs = [], {}
    for antigen, col in ANTIGENS.items():
        ts = nat[["ds", col]].rename(columns={col: "y"}).dropna()
        fc = prophet_fit(ts)
        cutoff = ts["ds"].max()
        base = ts[ts["ds"].dt.year == 2024]["y"].mean()
        fore = fc[fc["ds"] > cutoff].copy()
        fore["pct"] = fore["yhat"] / base * 100
        fore["ahead"] = (fore["ds"].dt.year - cutoff.year) * 12 + (fore["ds"].dt.month - cutoff.month)
        win = fore[(fore.ahead >= WINDOW[0]) & (fore.ahead <= WINDOW[1])]
        fcs[antigen] = dict(obs=ts, fc=fc, base=base)
        rows.append(dict(antigen=antigen, base_2024_doses_per_month=round(base), min_forecast_pct=round(fore.pct.min(), 1),
                         min_when=fore.loc[fore.pct.idxmin(), "ds"].strftime("%b %Y"),
                         crosses_80_in_6_12m=bool((win.pct < THRESHOLD).any()),
                         min_pct_in_window=round(win.pct.min(), 1)))
    return pd.DataFrame(rows), fcs, nat


def fit_lga(lga, sub):
    out, skipped = [], []
    if len(sub) < 18:
        return out, [dict(lga=lga, antigen=a, reason="fewer than 18 monthly rows") for a in ANTIGENS]
    state, zone = sub["state"].iloc[0], sub["zone"].iloc[0]
    for antigen, col in ANTIGENS.items():
        ts = sub[["ds", col]].rename(columns={col: "y"}).dropna()
        if ts["y"].sum() == 0 or len(ts) < 18:
            skipped.append(dict(lga=lga, antigen=antigen, reason="no doses or fewer than 18 months")); continue
        base = ts.loc[ts["ds"].dt.year == 2024, "y"].mean()
        if not np.isfinite(base) or base <= 0:
            skipped.append(dict(lga=lga, antigen=antigen, reason="no 2024 baseline")); continue
        fc = prophet_fit(ts)
        cutoff = ts["ds"].max()
        fore = fc[fc["ds"] > cutoff].copy()
        fore["yhat_pct"] = fore["yhat"] / base * 100
        fore["months_ahead"] = (fore["ds"].dt.year - cutoff.year) * 12 + (fore["ds"].dt.month - cutoff.month)
        win = fore[(fore.months_ahead >= 6) & (fore.months_ahead <= 12)]
        if win.empty:
            skipped.append(dict(lga=lga, antigen=antigen, reason="no forecast months in window")); continue
        below = fore[fore.yhat_pct < THRESHOLD]
        out.append(dict(state=state, zone=zone, lga=lga, antigen=antigen, baseline_2024_doses_per_month=round(base, 1),
                        min_forecast_pct_6_12m=round(win.yhat_pct.min(), 1),
                        min_forecast_when=win.loc[win.yhat_pct.idxmin(), "ds"].strftime("%Y-%m"),
                        crosses_80_in_6_12m=bool((win.yhat_pct < THRESHOLD).any()),
                        first_month_below_80=below["ds"].min().strftime("%Y-%m") if len(below) else None,
                        months_observed=len(ts), last_observed=cutoff.strftime("%Y-%m")))
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse", choices=["original", "corrected"], required=True)
    ap.add_argument("--jobs", type=int, default=20)
    a = ap.parse_args()
    out = ROOT / "outputs" / a.parse
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = load(a.parse)
    print(f"[{a.parse}] rows {len(d):,} | LGAs {d.lga.nunique()} | missing Penta1 cells {int(d.penta_1_count.isna().sum()):,}")
    nat_sum, fcs, nat = national(d)
    nat_sum.to_csv(out / "d1_national_at_risk_summary.csv", index=False)
    nat.to_csv(out / "d1_national_monthly_doses.csv", index=False)
    for k, v in fcs.items():
        v["fc"][["ds", "yhat", "yhat_lower", "yhat_upper"]].assign(antigen=k, base_2024=v["base"]).to_csv(out / f"d1_national_forecast_{k}.csv", index=False)
    print(nat_sum.to_string(index=False))
    lga_agg = d.groupby(["state", "zone", "lga", "ds"])[list(ANTIGENS.values())].sum().reset_index().sort_values(["state", "lga", "ds"])
    groups = [(lga, sub) for lga, sub in lga_agg.groupby("lga", sort=False)]
    res = Parallel(n_jobs=a.jobs, verbose=0)(delayed(fit_lga)(lga, sub) for lga, sub in groups)
    rows = [r for o, _ in res for r in o]
    skipped = pd.DataFrame([s for _, sk in res for s in sk])
    fc_df = pd.DataFrame(rows)
    at_risk = fc_df[fc_df.crosses_80_in_6_12m].sort_values(["antigen", "min_forecast_pct_6_12m"]).reset_index(drop=True)
    fc_df.to_csv(out / "d1_lga_forecast_all.csv", index=False)
    at_risk.to_csv(out / "d1_lga_at_risk_below_80pct.csv", index=False)
    skipped.to_csv(out / "d1_lga_not_forecast.csv", index=False)
    sev = pd.cut(at_risk.min_forecast_pct_6_12m, [-np.inf, 0, 40, 70, 80], labels=["Collapsed (<0%)", "Critical (0-40%)", "Severe (40-70%)", "Warning (70-80%)"])
    summ = dict(parse=a.parse, lga_antigen_series=len(groups) * 4, forecasts_fitted=len(fc_df), lgas_with_any_forecast=int(fc_df.lga.nunique()),
                lgas_with_all_four=int((fc_df.groupby("lga").size() == 4).sum()), alerts=len(at_risk),
                lgas_with_any_alert=int(at_risk.lga.nunique()), share_forecasts_flagged=round(len(at_risk) / max(len(fc_df), 1) * 100, 1),
                alerts_by_antigen=at_risk.antigen.value_counts().to_dict(), severity=sev.value_counts().to_dict(),
                forecasts_below_zero=int((fc_df.min_forecast_pct_6_12m < 0).sum()), seconds=round(time.time() - t0))
    (out / "d1_summary.json").write_text(json.dumps(summ, indent=2, default=int))
    print(json.dumps(summ, indent=2, default=int))


if __name__ == "__main__":
    main()
