"""Execute a built notebook end to end against its datasets folder (as a Colab user would), save executed copy."""
import os
import sys
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"; NB_DIR.mkdir(exist_ok=True)
DATA = ROOT / "Updated Datasets for Domain 5"
MAP = {"D5_Method1_Bayesian_Hierarchical_Model_LGA_Allocation.ipynb": "Method1_Bayesian_Hierarchical_Model",
       "D5_Method2_Bayesian_Small_Area_Estimation.ipynb": "Method2_Bayesian_Small_Area_Estimation",
       "D5_LGA_Archetypes_Unsupervised_Clustering.ipynb": "LGA_Archetypes"}

name = sys.argv[1]
src = Path(__file__).parent / name
nb = nbf.read(src, as_version=4)
os.environ["D5_DATA_DIR"] = str(DATA / MAP[name])
os.environ["D5_OUT_DIR"] = str(NB_DIR / ("outputs_" + MAP[name]))
t0 = time.time()
try:
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
    status = "EXECUTED_OK"
except Exception as e:  # noqa: BLE001
    status = "FAILED: " + str(e)[-2500:]
nbf.write(nb, NB_DIR / name)
print(name, status, f"{(time.time() - t0) / 60:.1f} min")
for c in nb.cells:
    if c.cell_type == "code":
        for o in c.get("outputs", []):
            if o.get("name") == "stdout":
                print(o["text"][-1500:])
