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
    # Multicollinearity: predictor pairs with |r| >= 0.8 (candidates to drop from a model).
    preds = [c for c in cols if c != OUTCOME]
    cp = df[preds].corr()
    pairs = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            r = float(cp.iloc[i, j])
            if abs(r) >= 0.8:
                # Suggest keeping the predictor more correlated with the outcome.
                keep = preds[i] if abs(corr.loc[preds[i], OUTCOME]) >= abs(corr.loc[preds[j], OUTCOME]) else preds[j]
                drop = preds[j] if keep == preds[i] else preds[i]
                pairs.append({"var_1": pretty(preds[i]), "var_2": pretty(preds[j]), "r": round(r, 2),
                              "suggest_keep": pretty(keep), "suggest_drop": pretty(drop)})
    pairs.sort(key=lambda d: -abs(d["r"]))
    summary = {"top_drivers_vs_zero_dose": {pretty(k): round(float(v), 2) for k, v in top.head(6).items()},
               "multicollinear_pairs_|r|>=0.8": pairs}
    return style_fig(fig), summary


def hist_box_fig(df: pd.DataFrame, col: str):
    y = df[col].dropna().astype(float)
    q0, q1, q2, q3, q4 = (float(np.percentile(y, p)) for p in (0, 25, 50, 75, 100))
    mean = float(y.mean())
    fig = make_subplots(rows=1, cols=2, column_widths=[0.66, 0.34], horizontal_spacing=0.10,
                        subplot_titles=(f"Distribution of {pretty(col)}", "Boxplot (5-number summary)"))
    fig.add_trace(go.Histogram(x=y, nbinsx=12, opacity=0.92, name="states",
                               marker=dict(color=C.STEEL, line=dict(color="white", width=1))),
                  row=1, col=1)
    fig.add_vline(x=mean, line=dict(color=C.ACCENT, width=2, dash="dash"), row=1, col=1,
                  annotation_text=f"mean {mean:.1f}", annotation_position="top")
    fig.add_trace(go.Box(y=y, name=pretty(col), boxmean="sd", boxpoints="all", jitter=0.5,
                         pointpos=-1.7, marker=dict(color=C.NAVY, size=5),
                         fillcolor="rgba(31,59,87,0.12)", line=dict(color=C.NAVY)), row=1, col=2)
    for val, lab in [(q4, "max"), (q3, "Q3"), (q2, "median"), (q1, "Q1"), (q0, "min")]:
        fig.add_annotation(x=0.99, y=val, xref="x2 domain", yref="y2", xanchor="right",
                           showarrow=False, text=f"{lab}: {val:.1f}",
                           font=dict(size=10, color=C.NAVY), bgcolor="rgba(255,255,255,0.82)")
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(title=f"{pretty(col)} - univariate distribution", showlegend=False,
                      height=440, bargap=0.06)
    stats = {"mean": round(mean, 1), "median": round(q2, 1), "Q1": round(q1, 1), "Q3": round(q3, 1),
             "IQR": round(q3 - q1, 1), "min": round(q0, 1), "max": round(q4, 1),
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


def band_bar_fig(df: pd.DataFrame):
    """Stacked bar of state counts per zero-dose burden band, by zone (replaces the Sankey)."""
    d = df[["zone_name", OUTCOME]].dropna().copy()
    d["band"] = _band(d[OUTCOME])
    zones = [z for z in ZONE_ORDER if z in set(d["zone_name"])]
    bands = ["Low (<20%)", "Moderate (20-40%)", "High (>40%)"]
    band_color = {"Low (<20%)": "#1A9850", "Moderate (20-40%)": "#FDAE61", "High (>40%)": "#D73027"}
    ct = (d.groupby(["zone_name", "band"], observed=True).size()
          .unstack(fill_value=0).reindex(index=zones, columns=bands, fill_value=0))
    fig = go.Figure()
    for b in bands:
        vals = ct[b].tolist()
        fig.add_trace(go.Bar(x=zones, y=vals, name=b, marker_color=band_color[b],
                             text=[v if v else "" for v in vals], textposition="inside"))
    fig.update_layout(barmode="stack", title="States per zero-dose burden band, by zone",
                      xaxis_title="Geopolitical zone", yaxis_title="Number of states",
                      legend=dict(orientation="h", y=-0.18), height=460)
    summary = {"states_per_band": d["band"].astype(str).value_counts().to_dict(),
               "high_band_zones": ct["High (>40%)"][ct["High (>40%)"] > 0].to_dict()}
    return style_fig(fig), summary


def beta_drivers(df: pd.DataFrame, top_k: int = 4):
    """Parsimonious Beta regression of the 2024 ZERO-DOSE rate on the top-k LASSO-selected drivers.

    LASSO selects on the logit scale (matching the Beta logit link); the top-k drivers by magnitude
    are then fit with a Beta (or fractional-logit fallback) model reporting the coefficient, its
    DIRECTION and a 95% confidence interval - uncertainty quantification, not a significance verdict.
    Returns (table, method_label, n_states)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV
    import stats_infer as si

    feats = [c for c in DRIVERS if c in df.columns]
    sub = df[[OUTCOME] + feats].dropna()
    if len(sub) < 12 or not feats:
        return None, "", 0
    p = np.clip(sub[OUTCOME].values / 100.0, 1e-3, 1 - 1e-3)
    Xz = pd.DataFrame(StandardScaler().fit_transform(sub[feats]), columns=feats)
    logit = np.log(p / (1 - p))
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(Xz.values, logit)
    coefs = dict(zip(feats, lasso.coef_))
    ranked = sorted(feats, key=lambda f: abs(coefs[f]), reverse=True)
    selected = [f for f in ranked if abs(coefs[f]) > 0][:top_k] or ranked[:top_k]
    tbl, method = si.beta_regression(p, Xz[selected])
    tbl = tbl[tbl["term"] != "(intercept)"].copy()
    tbl = si.add_ci(tbl, "coef", "SE")
    tbl["Driver"] = tbl["term"].map(pretty)
    tbl["Coef (logit)"] = tbl["coef"].round(2)
    tbl["95% CI"] = tbl.apply(lambda r: f"{r['CI_low']:.2f} to {r['CI_high']:.2f}", axis=1)
    tbl["Direction"] = tbl["coef"].map(lambda c: "Higher zero-dose" if c > 0 else "Lower zero-dose")
    tbl["abs"] = tbl["coef"].abs()
    out = (tbl.sort_values("abs", ascending=False)
           [["Driver", "Coef (logit)", "95% CI", "Direction"]].reset_index(drop=True))
    return out, method, len(sub)


def multiple_regression(df: pd.DataFrame, outcome: str, predictors: list[str]):
    """Build-your-own multiple linear regression: HC3 robust SEs, p-values, 95% CIs and model fit
    (R-squared, adjusted R-squared, overall F-test). Returns {table, stats} or None if too few rows."""
    import stats_infer as si
    from scipy import stats as ss
    preds = [p for p in predictors if p in df.columns and p != outcome]
    if not preds:
        return None
    sub = df[[outcome] + preds].apply(pd.to_numeric, errors="coerce").dropna()
    n, k = len(sub), len(preds)
    if n < k + 2:
        return None
    y = sub[outcome].values.astype(float)
    X = sub[preds]
    tbl = si.add_ci(si.ols_robust(y, X), "coef", "robust_SE")
    Xmat = np.column_stack([np.ones(n), X.values.astype(float)])
    beta = np.linalg.pinv(Xmat.T @ Xmat) @ Xmat.T @ y
    yhat = Xmat @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if (n - k - 1) > 0 else r2
    f_p = None
    if (1 - r2) > 0 and (n - k - 1) > 0:
        fstat = (r2 / k) / ((1 - r2) / (n - k - 1))
        f_p = round(float(ss.f.sf(fstat, k, n - k - 1)), 4)
    tbl["Term"] = tbl["term"].map(lambda t: "(intercept)" if t == "(intercept)" else pretty(t))
    tbl["Coefficient"] = tbl["coef"].round(3)
    tbl["Robust SE"] = tbl["robust_SE"].round(3)
    tbl["p-value"] = tbl["p_value"].round(4)
    tbl["95% CI"] = tbl.apply(lambda r: f"{r['CI_low']:.2f} to {r['CI_high']:.2f}", axis=1)
    table = tbl[["Term", "Coefficient", "Robust SE", "p-value", "95% CI"]].reset_index(drop=True)
    stats = {"n": n, "predictors": k, "R2": round(r2, 3), "adj_R2": round(adj, 3), "F_p": f_p}
    return {"table": table, "stats": stats}


def all_numeric(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("state_name", "zone_name")
            and pd.api.types.is_numeric_dtype(df[c])]


def _as_categorical(series: pd.Series):
    """Return a categorical version of a column (tertile-band numerics; pass strings through)."""
    if pd.api.types.is_numeric_dtype(series):
        try:
            return pd.qcut(series, 3, labels=["Low", "Mid", "High"], duplicates="drop").astype(str)
        except Exception:
            return pd.cut(series, 3, labels=["Low", "Mid", "High"]).astype(str)
    return series.astype(str)


def hypothesis_test(df: pd.DataFrame, test: str, **kw) -> dict:
    """Run a chosen statistical test and return a result dict (statistic, p, detail, question)."""
    from scipy import stats as ss
    alpha = 0.05
    if test == "Independent t-test":
        outcome, grp = kw["outcome"], kw["group_var"]
        med = df[grp].median()
        a = df[df[grp] <= med][outcome].dropna()
        b = df[df[grp] > med][outcome].dropna()
        t, p = ss.ttest_ind(a, b, equal_var=False)
        res = {"statistic_name": "t", "statistic": round(float(t), 3), "p_value": round(float(p), 4),
               "detail": {f"low {pretty(grp)} (n={len(a)}) mean {pretty(outcome)}": round(float(a.mean()), 1),
                          f"high {pretty(grp)} (n={len(b)}) mean {pretty(outcome)}": round(float(b.mean()), 1)},
               "question": f"Does {pretty(outcome)} differ between states with low vs high {pretty(grp)} "
                           "(split at the median)?"}
    elif test == "Paired t-test":
        a, b = kw["col_a"], kw["col_b"]
        d = df[[a, b]].dropna()
        t, p = ss.ttest_rel(d[a], d[b])
        res = {"statistic_name": "t", "statistic": round(float(t), 3), "p_value": round(float(p), 4),
               "detail": {f"mean {pretty(a)}": round(float(d[a].mean()), 1),
                          f"mean {pretty(b)}": round(float(d[b].mean()), 1),
                          "mean paired difference": round(float((d[b] - d[a]).mean()), 1), "n": len(d)},
               "question": f"Do {pretty(a)} and {pretty(b)} differ across the same states (paired)?"}
    elif test == "One-way ANOVA":
        outcome, grp = kw["outcome"], kw.get("group_var", "zone_name")
        groups = [g[outcome].dropna() for _, g in df.groupby(grp) if len(g[outcome].dropna()) > 1]
        f, p = ss.f_oneway(*groups)
        res = {"statistic_name": "F", "statistic": round(float(f), 3), "p_value": round(float(p), 4),
               "detail": {str(k): round(float(g[outcome].mean()), 1) for k, g in df.groupby(grp)},
               "question": f"Does mean {pretty(outcome)} differ across {pretty(grp)}?"}
    elif test == "Chi-square test":
        v1, v2 = kw["var1"], kw["var2"]
        ct = pd.crosstab(_as_categorical(df[v1]), _as_categorical(df[v2]))
        chi2, p, dof, _ = ss.chi2_contingency(ct)
        res = {"statistic_name": "chi-square", "statistic": round(float(chi2), 3),
               "p_value": round(float(p), 4), "dof": int(dof),
               "detail": ct.to_dict(),
               "question": f"Are {pretty(v1)} and {pretty(v2)} associated (numeric variables are "
                           "split into Low/Mid/High thirds)?"}
    else:
        return {"error": "Unknown test"}
    res["test"] = test
    res["alpha"] = alpha
    res["significant"] = bool(res["p_value"] < alpha)
    res["conclusion"] = ("Statistically significant at alpha 0.05 (reject the null hypothesis)."
                         if res["significant"] else
                         "Not statistically significant at alpha 0.05 (fail to reject the null).")
    return res


def bland_altman_fig(df: pd.DataFrame, a: str = "dtp1_2018", b: str = "dtp1_2024"):
    d = df[[a, b, "state_name"]].dropna()
    mean = (d[a] + d[b]) / 2
    diff = d[b] - d[a]
    bias, sd, n = float(diff.mean()), float(diff.std()), int(len(diff))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    within = int(((diff >= lo) & (diff <= hi)).sum())
    fig = go.Figure()
    fig.add_hrect(y0=lo, y1=hi, fillcolor="rgba(46,110,142,0.08)", line_width=0)
    fig.add_trace(go.Scatter(x=mean, y=diff, mode="markers", text=d["state_name"],
                             marker=dict(size=9, color=C.STEEL, line=dict(color="white", width=0.6)),
                             hovertemplate="%{text}<br>mean %{x:.1f}, diff %{y:.1f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#94A3B8", width=1.2, dash="dot"),
                  annotation_text="no difference", annotation_position="bottom right")
    for yv, lab, col in [(bias, f"mean bias {bias:.1f}", C.NAVY),
                         (lo, f"lower LoA {lo:.1f}", C.ACCENT), (hi, f"upper LoA {hi:.1f}", C.ACCENT)]:
        fig.add_hline(y=yv, line=dict(color=col, width=1.6, dash="dash"),
                      annotation_text=lab, annotation_position="right")
    fig.update_layout(title=f"Bland-Altman agreement: {pretty(a)} vs {pretty(b)}",
                      xaxis_title="Mean of the two measures (%)",
                      yaxis_title=f"Difference ({pretty(b)} - {pretty(a)}, % points)", height=480)
    return style_fig(fig), {"bias": round(bias, 1), "sd_of_diff": round(sd, 1),
                            "lower_loa": round(lo, 1), "upper_loa": round(hi, 1),
                            "within_95pct_loa": f"{within}/{n}"}
