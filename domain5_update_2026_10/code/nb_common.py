"""Shared notebook-building helpers (setup cell, name harmonisation code inlined into notebooks)."""
import nbformat as nbf


def md(t):
    return nbf.v4.new_markdown_cell(t.strip("\n"))


def code(t):
    return nbf.v4.new_code_cell(t.strip("\n"))


def setup_cell(required, pip_pkgs):
    return code(f'''
# ---- Setup: works in Google Colab and locally ------------------------------------------------
import os, sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REQUIRED = {required!r}

if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + {pip_pkgs!r}, check=False)
    DATA_DIR = Path("/content/data"); DATA_DIR.mkdir(exist_ok=True)
    # Option A (default): upload the files from the matching folder of "Updated Datasets for Domain 5".
    # Option B: mount Google Drive and point DATA_DIR at the folder, e.g.
    #   from google.colab import drive; drive.mount("/content/drive")
    #   DATA_DIR = Path("/content/drive/MyDrive/Updated Datasets for Domain 5/<folder>")
    missing = [f for f in REQUIRED if not (DATA_DIR / f).exists()]
    if missing:
        from google.colab import files
        print("Please select these files (you can select them all at once):", missing)
        for name, content in files.upload().items():
            (DATA_DIR / name).write_bytes(content)
else:
    DATA_DIR = Path(os.environ.get("D5_DATA_DIR", ".")).resolve()

OUT_DIR = Path(os.environ.get("D5_OUT_DIR", "outputs")).resolve(); OUT_DIR.mkdir(parents=True, exist_ok=True)
missing = [f for f in REQUIRED if not (DATA_DIR / f).exists()]
assert not missing, f"Missing input files in {{DATA_DIR}}: {{missing}}"
print("Data folder:", DATA_DIR.name, "| all", len(REQUIRED), "input files found | outputs ->", OUT_DIR.name)
''')


NAMES_CODE = r'''
# ---- LGA / state name harmonisation (DHIS2, NPC population file, GRID3 and IHME spell names differently)
import re, difflib
import numpy as np, pandas as pd

def clean_lga_name(s):
    """Remove the 2-letter DHIS2 state prefix and the 'Local Government Area' suffix."""
    s = str(s).strip()
    s = re.sub(r"^[a-z]{2}\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*local\s+government\s+area\s*$", "", s, flags=re.IGNORECASE)
    return s.strip().title()

def nstate(s):
    s = str(s).lower().strip()
    if "federal capital" in s or s in ("fct", "fct, abuja"):
        return "fct"
    return re.sub(r"\(.*?\)", "", s).strip()

def nlga(s):
    s = re.sub(r"\(.*?\)|\[.*?\]", "", str(s).lower().strip())
    return re.sub(r"\s+", " ", s.replace("'", "").replace("/", " ").replace("-", " ").replace(".", "")).strip()

def tok(s):
    return " ".join(sorted(nlga(s).split()))

LGA_ALIAS = {("kebbi", "arewa"): "arewa dandi", ("imo", "ezinihitte mbaise"): "ezinihitte",
             ("ekiti", "aiyekire"): "gbonyin"}
FCT6 = {"Abaji", "Abuja Municipal Area Council", "Bwari", "Gwagwalada", "Kuje", "Kwali"}

def make_matcher(keys_by_state):
    """Return f(state_key, lga) -> matched key: exact, alias, token-sorted, then close match (cutoff 0.75)."""
    def match(sk, lga):
        l = nlga(lga); cands = keys_by_state.get(sk, [])
        if l in cands: return l, "exact"
        a = LGA_ALIAS.get((sk, l))
        if a in cands: return a, "alias"
        tk = {tok(c): c for c in cands}
        if tok(l) in tk: return tk[tok(l)], "token"
        c = difflib.get_close_matches(l, cands, n=1, cutoff=0.75)
        return (c[0], "close") if c else (None, "unmatched")
    return match
'''
