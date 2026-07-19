"""
Isolated worker process for the heavy native model calls.

Invoked by isolation.run_isolated() as:
    python model_worker.py <job> <input.pkl> <output.pkl>

It runs one job, writes the pickled result, and exits. If the native maths libraries fault
(SIGSEGV), only THIS process dies - the parent Streamlit app stays up and reports the failure.
The computation itself is unchanged, so results are identical to running it in-process.
"""
import os
import pickle
import sys

# Keep the native thread pools bounded inside the child too, matching the container budget.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
             "RAYON_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    job, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(inp, "rb") as fh:
        payload = pickle.load(fh)

    if job == "state":
        from models.d5_zerodose import _fit_state_live
        result = _fit_state_live(**payload)
    else:
        # Only the Bayesian fit is isolated. The Gi* hotspot path stays in-process: it is
        # guarded by precomputed results for the bundled data, and isolating it caused the
        # worker to hang (geopandas/esda import under a non-Streamlit runtime).
        raise ValueError(f"unknown job: {job}")

    with open(outp, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
