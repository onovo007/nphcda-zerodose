"""One-time: run the Domain 5 Bayesian state model + LGA burden on the BUNDLED sample data and save
the exact outputs, so the deployed app loads them instead of sampling live (which segfaults native
BLAS on small shared containers). Uploaded data still runs live. Re-run only if the sample data or
model changes."""
import warnings, json
warnings.filterwarnings("ignore")
import config as C
import data_io as io
from models.d5_zerodose import (run_state_model, run_lga_burden, _fp_state, _fp_lga, _PRECOMP)

_PRECOMP.mkdir(parents=True, exist_ok=True)
data = io.load_sample()

print("Sampling state model (bundled sample, live)...")
so = run_state_model(data["ndhs_long"], data["under5"], data["dhis2"],
                     key="precompute", draws=C.MCMC_DRAWS_LIVE, tune=C.MCMC_TUNE_LIVE)
so["res"].to_parquet(_PRECOMP / "state_res.parquet")
so["diag"].to_parquet(_PRECOMP / "state_diag.parquet")
(_PRECOMP / "state_meta.json").write_text(json.dumps({
    "fp": _fp_state(data["ndhs_long"]), "max_rhat": so["max_rhat"],
    "min_ess": so["min_ess"], "n_draws": so["n_draws"]}))
print(f"  state: {len(so['res'])} states | national 2026 = {so['res']['zd_count_2026'].sum():,.0f} "
      f"| maxRhat={so['max_rhat']} | fp={_fp_state(data['ndhs_long'])}")

print("Computing LGA burden...")
lo = run_lga_burden(data["dhis2"], so["res"], data["lga_population"], key="precompute")
lo["clean"].to_parquet(_PRECOMP / "lga_clean.parquet")
lo["pareto"].to_parquet(_PRECOMP / "lga_pareto.parquet")
(_PRECOMP / "lga_stats.json").write_text(json.dumps({
    "fp": _fp_lga(data["dhis2"], so["res"]),
    "national_total": int(lo["national_total"]), "n_lgas": int(lo["n_lgas"]),
    "top20_pct": lo["top20_pct"], "n80": int(lo["n80"]), "matched_pop": int(lo["matched_pop"])}))
print(f"  lga: {lo['n_lgas']} LGAs | total = {lo['national_total']:,} | top20%={lo['top20_pct']}")

print("Computing Getis-Ord Gi* (LGA + per forecast year state)...")
import names as N
import spatial
from spatial import _fp_gi_lga, _fp_gi_state
clean_df = lo["clean"].rename(columns={"State": "state", "LGA": "lga", "ZD proxy (%)": "zd_proxy_pct"}
                              )[["state", "lga", "zd_proxy_pct"]]
gi = spatial.lga_gi_star(clean_df, key="precompute")
gi.to_parquet(_PRECOMP / "lga_gi.parquet")
gi_meta = {"lga_fp": _fp_gi_lga(clean_df), "state": {}}
res_keyed = so["res"].copy()
res_keyed["state_key"] = res_keyed["state"].map(N.nstate)
for yr in C.FORECAST_YEARS:
    vcol = f"zd_pred_{yr}_mean"
    if vcol not in res_keyed.columns:
        continue
    sv = res_keyed[["state_key", vcol]].rename(columns={vcol: "value"})
    sg = spatial.state_gi_star(sv, key=f"pc-{yr}", value_col="value")
    fn = f"state_gi_{yr}.parquet"
    sg.to_parquet(_PRECOMP / fn)
    gi_meta["state"][_fp_gi_state(sv, "value")] = fn
(_PRECOMP / "gi_meta.json").write_text(json.dumps(gi_meta))
print(f"  gi: LGA hotspots {gi['gi_class'].value_counts().to_dict()} | state years {list(gi_meta['state'].values())}")
print("Saved precomputed results to", _PRECOMP)
