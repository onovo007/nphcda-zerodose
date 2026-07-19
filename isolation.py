"""
Process isolation for the heavy native model calls.

PyMC/nutpie sampling and esda's Getis-Ord Gi* run compiled native code (BLAS, OpenMP, LLVM,
Rust). When that native code faults it raises SIGSEGV, which Python cannot catch - the whole
Streamlit process dies and every user's session goes with it.

Running those calls in a CHILD process contains the blast radius: a fault kills only the child,
the parent sees a non-zero exit code and turns it into a normal Python exception the app can
report. Same code, same data, same seed -> identical results; only the process boundary changes.
(This is exactly why Prophet has never taken the app down - cmdstanpy already runs Stan this way.)
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

_WORKER = Path(__file__).with_name("model_worker.py")
DEFAULT_TIMEOUT = int(os.environ.get("MODEL_JOB_TIMEOUT", "1800"))  # 30 min ceiling


class IsolatedRunError(RuntimeError):
    """Raised when the isolated worker crashed, timed out, or returned nothing usable."""


def run_isolated(job: str, payload: dict, timeout: int = DEFAULT_TIMEOUT):
    """Run `job` in a child process with `payload` as kwargs; return its result.

    Raises IsolatedRunError on native crash (e.g. exit -11/139 = SIGSEGV), timeout or bad output.
    """
    with tempfile.TemporaryDirectory(prefix="modeljob_") as td:
        inp, outp = Path(td) / "in.pkl", Path(td) / "out.pkl"
        with open(inp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

        try:
            proc = subprocess.run(
                [sys.executable, str(_WORKER), job, str(inp), str(outp)],
                cwd=str(_WORKER.parent), capture_output=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise IsolatedRunError(
                f"The {job} model run exceeded {timeout // 60} minutes and was stopped. "
                "Try a smaller upload, or re-run when the server is less busy."
            ) from None

        if proc.returncode != 0 or not outp.exists():
            tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-6:]
            detail = " | ".join(tail) if tail else "no error output"
            # -11 (POSIX) / 139 (shell) both mean SIGSEGV in the native maths libraries.
            if proc.returncode in (-11, 139):
                raise IsolatedRunError(
                    f"The {job} model hit a low-level numerical fault on this dataset and was "
                    "stopped safely. The platform is still running - please check the uploaded "
                    f"file and try again. (worker exit {proc.returncode})"
                )
            raise IsolatedRunError(
                f"The {job} model run failed (exit {proc.returncode}). {detail}"
            )

        with open(outp, "rb") as fh:
            return pickle.load(fh)
