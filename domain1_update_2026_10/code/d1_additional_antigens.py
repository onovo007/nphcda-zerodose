"""Domain 1: additional-antigen forecasts (same method as D1_Additional_Antigens_Forecast.ipynb).

Main DHIS2 file merged with the additional-antigen file on zone/state/lga/period; national monthly sums; months with
zero national doses dropped; Prophet (yearly seasonality, interval_width 0.95) with a 30-month horizon; minimum forecast
over the horizon as % of the antigen's 2024 mean monthly doses. Established antigens are flagged "at risk of decline"
below 80%; recently introduced antigens are read as uptake.
--parse original | corrected (as in d1_forecast.py)
"""
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "All I Need" / "03_Datasets" / "datasets"
ESTABLISHED = {"OPV3": "opv_3_count", "IPV1": "ipv_1_count", "PCV3": "pcv_3_count", "Yellow Fever": "yellow_fever_count",
               "Men A": "men_a_count"}
RECENT = {"IPV2": "ipv_2_count", "Rota1": "rota_1_count", "Rota2": "rota_2_count", "Rota3": "rota_3_count"}


def load(parse):
    kw = {} if parse == "original" else dict(thousands=",", low_memory=False)
    old = pd.read_csv(DATA / "dhis2_data_all_states.csv", **kw)
    new = pd.read_csv(DATA / "dhis2_data_additional_antigens.csv", **kw)
    keys = ["zone", "state", "lga", "period"]
    df = old.merge(new.drop(columns=[c for c in ["row_id"] if c in new.columns]), on=keys, how="left")
    df["ds"] = pd.to_datetime(df["period"].astype(str).str.strip(), format="%b-%y", errors="coerce")
    return df


def forecast_one(df, col):
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    v = pd.to_numeric(df[col], errors="coerce")
    s = df.assign(y=v).dropna(subset=["ds"]).groupby("ds")["y"].sum().reset_index()
    s = s[s["y"] > 0].sort_values("ds")
    if len(s) < 12:
        return None, None, None
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95).fit(s)
    fc = m.predict(m.make_future_dataframe(periods=30, freq="MS"))
    base = s[s["ds"].dt.year == 2024]["y"].mean()
    return s, fc[fc["ds"] > s["ds"].max()], base


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parse", choices=["original", "corrected"], required=True)
    a = ap.parse_args()
    out = ROOT / "outputs" / a.parse; out.mkdir(parents=True, exist_ok=True)
    df = load(a.parse)
    rows, series = [], []
    for label, col in {**ESTABLISHED, **RECENT}.items():
        s, fcf, base = forecast_one(df, col)
        if s is None:
            continue
        kind = "established" if label in ESTABLISHED else "recently introduced"
        minpct = fcf["yhat"].min() / base * 100 if base else np.nan
        flag = "n/a - scaling up" if kind == "recently introduced" else ("AT-RISK-OF-DECLINE" if minpct < 80 else "on track")
        rows.append({"Antigen": label, "Group": kind, "2024 base doses/mo": round(base), "Min forecast % of 2024": round(minpct, 1),
                     "Month of minimum": fcf.loc[fcf["yhat"].idxmin(), "ds"].strftime("%b %Y"), "Early-warning": flag,
                     "2025 observed % of 2024": round(s[s.ds.dt.year == 2025].y.mean() / base * 100, 1)})
        series.append(pd.concat([s.assign(kind="observed"), fcf[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"yhat": "y"}).assign(kind="forecast")])
                      .assign(antigen=label, group=kind, base_2024=base))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "D1_additional_antigen_forecast_summary.csv", index=False)
    pd.concat(series).to_csv(out / "D1_additional_antigen_series.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
