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
print("Saved precomputed results to", _PRECOMP)
