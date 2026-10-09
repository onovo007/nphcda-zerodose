"""Single IHME DTP1 2018 LGA join used everywhere (same logic as in the Method 1 notebook)."""
import difflib
import re

import numpy as np
import pandas as pd


def nstate(s):
    s = str(s).lower().strip()
    if "federal capital" in s or s in ("fct", "fct, abuja"):
        return "fct"
    return re.sub(r"\(.*?\)", "", s).strip()


def nlga(s):
    s = re.sub(r"\(.*?\)|\[.*?\]", "", str(s).lower().strip())
    return re.sub(r"\s+", " ", s.replace("'", "").replace("/", " ").replace("-", " ").replace(".", "")).strip()


ALIAS = {("kebbi", "arewa"): "arewa dandi", ("imo", "ezinihitte mbaise"): "ezinihitte"}


def ihme_zero_dose(ihme_csv, states, lgas):
    """Return IHME 2018 zero-dose % (100 - DTP1) for each (state, lga); NaN if unmatched."""
    ih = pd.read_csv(ihme_csv)
    ih["sk"] = ih.ADM1_NAME.map(nstate).replace({"nassarawa": "nasarawa", "abuja": "fct"})
    ih["lk"] = ih.ADM2_NAME.map(nlga)
    look = {(s, l): v for s, l, v in zip(ih.sk, ih.lk, ih.dtp1_2018)}
    by = ih.groupby("sk")["lk"].apply(list).to_dict()
    scale = 100 if ih.dtp1_2018.max() <= 1.5 else 1
    out = []
    for st, lg in zip(states, lgas):
        sk, l = nstate(st), nlga(lg)
        v = look.get((sk, l))
        if v is None and ALIAS.get((sk, l)):
            v = look.get((sk, ALIAS[(sk, l)]))
        if v is None:
            c = difflib.get_close_matches(l, by.get(sk, []), n=1, cutoff=0.80)
            v = look[(sk, c[0])] if c else None
        out.append(np.nan if v is None else 100 - v * scale)
    return np.array(out, dtype=float)
