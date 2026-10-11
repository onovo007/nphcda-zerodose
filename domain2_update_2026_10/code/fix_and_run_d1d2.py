"""Fix the D1_D2 notebook (DHIS2 thousands separators), then execute it end to end on the bundled inputs.

Changes to the notebook:
  1. Section 2 reads DHIS2 with thousands="," so counts written as "1,234" are kept (previously read as missing).
  2. Package installation runs only in Google Colab (local runs use the existing environment).
  3. A short guide at the top points to the dedicated Domain 1 and Domain 5 notebooks.
The original notebook is not modified; the fixed, executed copy is written to Domain2_Update_20261011/notebooks/.
"""
import shutil
import sys
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parents[1] / "nphcda-zerodose-git"
SRC = APP / "notebooks" / "D1_D2__domains_1_2_final.ipynb"
RUN = ROOT / "notebooks" / "run"
RUN.mkdir(parents=True, exist_ok=True)
for f in ("dhis2_data_all_states.csv", "nigeria_zero_dose_model_dataset.csv"):
    shutil.copy2(APP / "data" / "sample" / f, RUN / f)

nb = nbf.read(SRC, as_version=4)
c9 = nb.cells[9]
assert c9.source.startswith("d = pd.read_csv(DHIS2_PATH)")
c9.source = c9.source.replace("d = pd.read_csv(DHIS2_PATH)",
                              "# DHIS2 exports counts of 1,000 or more with a thousands separator (\"1,234\"); read them as numbers\n"
                              "d = pd.read_csv(DHIS2_PATH, thousands=\",\", low_memory=False)", 1)
c3 = nb.cells[3]
assert "subprocess.check_call" in c3.source
c3.source = ("import subprocess, sys\n\nif IN_COLAB:\n" + "\n".join("    " + l for l in c3.source.replace("import subprocess, sys\n", "")
                                                                    .strip("\n").split("\n"))
             + "\nelse:\n    print(\"Local run: using the installed environment.\")\n")
guide = nbf.v4.new_markdown_cell(
    "**Reading guide.** Domain 2 (dropout and completion) in this notebook uses DHIS2 counts read with thousands separators "
    "(\"1,234\"). For Domain 1, the LGA early-warning forecasts for all 773 LGAs are in "
    "`D1_Antigen_EarlyWarning_Forecasts_773_LGAs.ipynb`. For Domain 5, the LGA-level estimates and archetypes are in "
    "`D5_Method1_Bayesian_Hierarchical_Model_LGA_Allocation.ipynb`, `D5_Method2_Bayesian_Small_Area_Estimation.ipynb` and "
    "`D5_LGA_Archetypes_Unsupervised_Clustering.ipynb`; the state-level clustering in Section 7 here is the earlier "
    "proof of concept.")
nb.cells.insert(1, guide)
nb.metadata.setdefault("colab", {"provenance": []})

t0 = time.time()
try:
    NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(RUN)}}).execute()
    status = "EXECUTED_OK"
except Exception as e:  # noqa: BLE001
    status = "FAILED: " + str(e)[-3000:]
out = ROOT / "notebooks" / "D1_D2__domains_1_2_final.ipynb"
nbf.write(nb, out)
print(status, f"{(time.time() - t0) / 60:.1f} min ->", out)
sys.exit(0 if status == "EXECUTED_OK" else 1)
