"""
Implementation Science - exploratory data analysis of the state-level zero-dose dataset.

Univariate (describe, correlation matrix, histogram, boxplot), bivariate (scatter with
Pearson r and p, violin by zone with Kruskal-Wallis p, Sankey, mosaic) and a Bland-Altman
agreement plot. Built on the zero-dose model dataset (37 states x equity/socioeconomic
covariates + survey zero-dose outcomes). Pure graph_objects + scipy/statsmodels (pandas-3 safe).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as C
from theme import style_fig, clean

OUTCOME = "zero_dose_2024"
DRIVERS = [
    "anc_4plus", "delivered_in_hf", "pct_c12to23_vax_card_seen", "pct_cu5_birth_registered",
    "pct_using_improved_water_source", "pct_cu5_slept_under_itn", "pct_problem_accessing_hfdistance",
    "pct_cu5_stunted", "pct_women_no_education", "pct_lowest_wealth_quintile",
    "total_fertility_rate", "pct_urban", "pct_muslim", "pct_women_with_mobile_phone",
    "pct_media_at_least_once_week", "pct_women_curr_employed",
]
ZONE_ORDER = C.ZONE_ORDER


def prep(model_dataset: pd.DataFrame) -> pd.DataFrame:
    df = model_dataset.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if c not in ("state_name", "zone_name"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = [OUTCOME] + [c for c in DRIVERS if c in df.columns]
    return [c for c in cols if c in df.columns]


def pretty(name: str) -> str:
    return clean(name.replace("pct_", "").replace("_", " ").title())


# --------------------------------------------------------------------------------------
def describe_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_cols(df)
    out = df[cols].describe().T.round(1).reset_index().rename(columns={"index": "Variable"})
    out["Variable"] = out["Variable"].map(pretty)
    return out


def corr_fig(df: pd.DataFrame):
    cols = numeric_cols(df)
    corr = df[cols].corr().round(2)
    labels = [pretty(c) for c in cols]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}", textfont=dict(size=8),
        colorbar=dict(title="r")))
    fig.update_layout(title="Correlation matrix - zero-dose and its drivers",
                      height=640, xaxis=dict(tickangle=-45))
    top = corr[OUTCOME].drop(OUTCOME).reindex(corr[OUTCOME].drop(OUTCOME).abs().sort_values(
        ascending=False).index)
    summary = {pretty(k): float(v) for k, v in top.head(6).items()}
    return style_fig(fig), summary


def hist_box_fig(df: pd.DataFrame, col: str):
    y = df[col].dropna()
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=(f"Distribution of {pretty(col)}", "Boxplot"))
    fig.add_trace(go.Histogram(x=y, nbinsx=12, marker_color=C.STEEL, name="count"), row=1, col=1)
    fig.add_trace(go.Box(y=y, marker_color=C.NAVY, boxpoints="all", name=pretty(col)), row=1, col=2)
    fig.update_layout(title=f"{pretty(col)} - univariate distribution", showlegend=False, height=420)
    stats = {"mean": round(float(y.mean()), 1), "median": round(float(y.median()), 1),
             "min": round(float(y.min()), 1), "max": round(float(y.max()), 1),
             "std": round(float(y.std()), 1), "skew": round(float(y.skew()), 2)}
    return style_fig(fig), stats


def scatter_fig(df: pd.DataFrame, x: str, y: str = OUTCOME):
    from scipy import stats as sstats
    d = df[[x, y, "state_name"]].dropna()
    r, p = sstats.pearsonr(d[x], d[y])
    slope, intercept = np.polyfit(d[x], d[y], 1)
    xs = np.linspace(d[x].min(), d[x].max(), 50)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="markers", text=d["state_name"],
                             marker=dict(size=9, color=C.STEEL, line=dict(color="white", width=0.6)),
                             hovertemplate="%{text}<br>%{x:.1f}, %{y:.1f}<extra></extra>", name="states"))
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                             line=dict(color=C.ACCENT, width=2.4, dash="dash"), name="OLS fit"))
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False,
                       align="left", bgcolor="rgba(255,255,255,0.85)",
                       text=f"Pearson r = {r:.2f}<br>p = {p:.3f}",
                       font=dict(size=12, color=C.NAVY))
    fig.update_layout(title=f"{pretty(x)} vs zero-dose rate (2024)",
                      xaxis_title=pretty(x), yaxis_title="Zero-dose rate 2024 (%)", height=460)
    return style_fig(fig), {"r": round(float(r), 2), "p": round(float(p), 3),
                            "direction": "positive" if r > 0 else "negative"}


def violin_fig(df: pd.DataFrame, value: str = OUTCOME, group: str = "zone_name"):
    from scipy import stats as sstats
    zones = [z for z in ZONE_ORDER if z in set(df[group])]
    groups = [df[df[group] == z][value].dropna() for z in zones]
    fig = go.Figure()
    for z, g in zip(zones, groups):
        fig.add_trace(go.Violin(y=g, name=z, box_visible=True, meanline_visible=True,
                                points="all", marker=dict(size=5),
                                line_color=C.ZONE_COLORS.get(z, C.STEEL)))
    p = np.nan
    try:
        p = sstats.kruskal(*[g for g in groups if len(g) > 0]).pvalue
    except Exception:
        pass
    fig.update_layout(title=f"{pretty(value)} by geopolitical zone (Kruskal-Wallis p = "
                            f"{p:.3f})" if not np.isnan(p) else f"{pretty(value)} by zone",
                      yaxis_title="Zero-dose rate 2024 (%)", showlegend=False, height=460)
    meds = {z: round(float(g.median()), 1) for z, g in zip(zones, groups) if len(g)}
    return style_fig(fig), {"kruskal_p": round(float(p), 3) if not np.isnan(p) else None,
                            "zone_medians": meds}


def _band(s: pd.Series) -> pd.Series:
    return pd.cut(s, bins=[-0.1, 20, 40, 200],
                 labels=["Low (<20%)", "Moderate (20-40%)", "High (>40%)"])


def sankey_fig(df: pd.DataFrame):
    d = df[["zone_name", OUTCOME]].dropna().copy()
    d["band"] = _band(d[OUTCOME])
    zones = [z for z in ZONE_ORDER if z in set(d["zone_name"])]
    bands = ["Low (<20%)", "Moderate (20-40%)", "High (>40%)"]
    nodes = zones + bands
    idx = {n: i for i, n in enumerate(nodes)}
    ct = d.groupby(["zone_name", "band"], observed=True).size().reset_index(name="n")
    band_color = {"Low (<20%)": "#1A9850", "Moderate (20-40%)": "#FDAE61", "High (>40%)": "#D73027"}
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=16, thickness=16,
                  color=[C.ZONE_COLORS.get(z, C.STEEL) for z in zones] + [band_color[b] for b in bands]),
        link=dict(source=[idx[r["zone_name"]] for _, r in ct.iterrows()],
                  target=[idx[r["band"]] for _, r in ct.iterrows()],
                  value=[int(r["n"]) for _, r in ct.iterrows()],
                  color="rgba(46,110,142,0.35)")))
    fig.update_layout(title="Flow of states: zone to zero-dose burden band", height=460)
    counts = ct.groupby("band", observed=True)["n"].sum().to_dict()
    return style_fig(fig), {"states_per_band": {str(k): int(v) for k, v in counts.items()}}


def mosaic_fig(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.graphics.mosaicplot import mosaic
    d = df[["zone_name", OUTCOME]].dropna().copy()
    d["band"] = _band(d[OUTCOME]).astype(str)
    counts = d.groupby(["zone_name", "band"], observed=True).size().to_dict()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    try:
        mosaic(counts, ax=ax, gap=0.015, title="Zone x zero-dose band (mosaic)",
               labelizer=lambda k: "")
    except Exception:
        ax.text(0.5, 0.5, "Mosaic unavailable for this data", ha="center")
    fig.tight_layout()
    return fig


def bland_altman_fig(df: pd.DataFrame, a: str = "dtp1_2018", b: str = "dtp1_2024"):
    d = df[[a, b, "state_name"]].dropna()
    mean = (d[a] + d[b]) / 2
    diff = d[b] - d[a]
    bias = float(diff.mean())
    sd = float(diff.std())
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean, y=diff, mode="markers", text=d["state_name"],
                             marker=dict(size=9, color=C.STEEL, line=dict(color="white", width=0.6)),
                             hovertemplate="%{text}<br>mean %{x:.1f}, diff %{y:.1f}<extra></extra>"))
    for yv, lab, col in [(bias, f"bias {bias:.1f}", C.NAVY), (lo, f"-1.96 SD {lo:.1f}", C.ACCENT),
                         (hi, f"+1.96 SD {hi:.1f}", C.ACCENT)]:
        fig.add_hline(y=yv, line=dict(color=col, width=1.4, dash="dash"),
                      annotation_text=lab, annotation_position="right")
    fig.update_layout(title=f"Bland-Altman agreement: {pretty(a)} vs {pretty(b)}",
                      xaxis_title="Mean of the two measures", yaxis_title="Difference", height=460)
    return style_fig(fig), {"bias": round(bias, 1), "lower_loa": round(lo, 1), "upper_loa": round(hi, 1)}
