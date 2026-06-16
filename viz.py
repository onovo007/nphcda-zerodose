"""
Shared premium Plotly figure builders. Every chart mirrors a notebook figure (chart type,
colours, threshold lines) but is interactive. All titles pass through the house sanitiser.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config as C
from theme import style_fig, clean


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _add_band(fig, x, lo, hi, color, alpha, name, legend=True):
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1], y=list(hi) + list(lo)[::-1],
        fill="toself", fillcolor=_hex_rgba(color, alpha), line=dict(width=0),
        hoverinfo="skip", name=name, showlegend=legend,
    ))


# --------------------------------------------------------------------------------------
# Forecast with prediction bands (D1 antigen %-baseline, D2 dropout)
# --------------------------------------------------------------------------------------
def forecast_band_fig(s: dict, color: str, title: str, ylabel: str, *,
                      threshold: float | None = None, mark_below: bool = False,
                      ref_line: float | None = None, ref_label: str = "") -> go.Figure:
    ox = pd.to_datetime(s["obs_x"]); fx = pd.to_datetime(s["fore_x"]); hx = pd.to_datetime(s["hist_x"])
    fig = go.Figure()
    _add_band(fig, fx, s["lo95"], s["hi95"], color, 0.13, "95% PI")
    _add_band(fig, fx, s["lo80"], s["hi80"], color, 0.26, "80% PI")
    fig.add_trace(go.Scatter(x=ox, y=s["obs_y"], mode="markers", name="Observed",
                             marker=dict(size=6, color=color, opacity=0.65)))
    fig.add_trace(go.Scatter(x=hx, y=s["hist_y"], mode="lines", name="Fitted",
                             line=dict(color=color, width=2.4)))
    fig.add_trace(go.Scatter(x=fx, y=s["fore_y"], mode="lines", name="Forecast",
                             line=dict(color=color, width=2.6, dash="dash")))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=C.ACCENT, width=1.8, dash="dash"),
                      annotation_text=f"{threshold:.0f}% target", annotation_position="top left")
        if mark_below:
            fy = np.array(s["fore_y"])
            below = fy < threshold
            if below.any():
                fig.add_trace(go.Scatter(
                    x=fx[below], y=fy[below], mode="markers", name="Below target",
                    marker=dict(symbol="triangle-down", size=11, color=C.ACCENT,
                                line=dict(color="white", width=0.8))))
    else:
        fig.add_hline(y=0, line=dict(color="#333", width=1.2, dash="dot"))
    if ref_line is not None:
        fig.add_hline(y=ref_line, line=dict(color="#6A2C91", width=1.8, dash="dot"),
                      annotation_text=ref_label or f"survey {ref_line:.0f}%",
                      annotation_position="bottom left",
                      annotation_font=dict(color="#6A2C91"))
    fig.add_vline(x=pd.to_datetime(s["cutoff"]), line=dict(color="#888", width=1.1, dash="dot"))
    fig.update_layout(title=title, yaxis_title=ylabel, xaxis_title="Month",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.18))
    return style_fig(fig, height=420)


# --------------------------------------------------------------------------------------
# D1 EDA antigen counts
# --------------------------------------------------------------------------------------
def antigen_eda_fig(nat: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    items = [(a, c) for a, c in C.ANTIGEN_TS.items() if c in nat.columns]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[a for a, _ in items])
    for i, (a, col) in enumerate(items):
        r, cc = i // 2 + 1, i % 2 + 1
        color = C.ANTIGEN_PAL[a]
        fig.add_trace(go.Scatter(x=nat["ds"], y=nat[col] / 1e3, mode="lines", name=a,
                                 line=dict(color=color, width=2.2), fill="tozeroy",
                                 fillcolor=_hex_rgba(color, 0.18), showlegend=False), row=r, col=cc)
        ma = nat[col].rolling(3, center=True).mean() / 1e3
        fig.add_trace(go.Scatter(x=nat["ds"], y=ma, mode="lines", showlegend=False,
                                 line=dict(color=color, width=1.3, dash="dash")), row=r, col=cc)
    fig.update_layout(title="National monthly antigen dose counts (thousands)")
    return style_fig(fig, height=560)


# --------------------------------------------------------------------------------------
# D2 LASSO driver bars
# --------------------------------------------------------------------------------------
def lasso_bars_fig(coefs: pd.Series, label: str, color: str) -> go.Figure:
    top = coefs.head(12).iloc[::-1]
    names = [clean(f.replace("pct_", "").replace("_", " ").title()) for f in top.index]
    fig = go.Figure(go.Bar(
        x=top.values, y=names, orientation="h",
        marker=dict(color=color, line=dict(color="white", width=0.5)),
        text=[f"{v:.3f}" for v in top.values], textposition="outside",
    ))
    fig.update_layout(title=f"{label} dropout - LASSO drivers",
                      xaxis_title="|LASSO coefficient|", yaxis_title="")
    return style_fig(fig, height=420)


# --------------------------------------------------------------------------------------
# D2 dropout heatmap (state x year)
# --------------------------------------------------------------------------------------
def dropout_heatmap_fig(piv: pd.DataFrame, title: str, last_obs_year: int) -> go.Figure:
    cols = [int(c) for c in piv.columns]
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=[str(c) for c in cols], y=piv.index.tolist(),
        colorscale="RdYlGn_r", zmid=0, zmin=-30, zmax=30,
        colorbar=dict(title="Dropout (%)"), xgap=1, ygap=1,
        hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
    ))
    fc_cols = [c for c in cols if c > last_obs_year]
    if fc_cols:
        n_hist = sum(1 for c in cols if c <= last_obs_year)
        fig.add_vline(x=n_hist - 0.5, line=dict(color="#0F172A", width=2.4))
    fig.update_layout(title=title, height=760, yaxis=dict(autorange="reversed"),
                      xaxis_title="", yaxis_title="State")
    return style_fig(fig)


# --------------------------------------------------------------------------------------
# D5 state trajectories
# --------------------------------------------------------------------------------------
def state_trajectories_fig(res: pd.DataFrame, top_n: int = 20) -> go.Figure:
    d = res.sort_values("zd_obs_2024", ascending=False).head(top_n)
    fig = go.Figure()
    for _, row in d.iterrows():
        color = C.ZONE_COLORS.get(row["zone"], C.MUTE)
        obs_x = C.NDHS_YEARS
        obs_y = [row[f"zd_obs_{y}"] for y in C.NDHS_YEARS]
        fc_x = [2024] + C.FORECAST_YEARS
        fc_y = [row["zd_obs_2024"]] + [row[f"zd_pred_{y}_mean"] for y in C.FORECAST_YEARS]
        lo = [row["zd_obs_2024"]] + [row[f"zd_pred_{y}_lo95"] for y in C.FORECAST_YEARS]
        hi = [row["zd_obs_2024"]] + [row[f"zd_pred_{y}_hi95"] for y in C.FORECAST_YEARS]
        _add_band(fig, fc_x, lo, hi, color, 0.07, "", legend=False)
        fig.add_trace(go.Scatter(x=obs_x, y=obs_y, mode="lines+markers", name=row["state"],
                                 legendgroup=row["zone"], line=dict(color=color, width=1.8),
                                 marker=dict(size=5),
                                 hovertemplate=f"{row['state']}<br>%{{x}}: %{{y:.1f}}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=fc_x, y=fc_y, mode="lines", showlegend=False,
                                 legendgroup=row["zone"], line=dict(color=color, width=1.8, dash="dash"),
                                 hoverinfo="skip"))
    fig.add_vline(x=2024, line=dict(color="#444", width=1.2, dash="dot"),
                  annotation_text="Forecast", annotation_position="top right")
    fig.update_layout(title=f"State zero-dose trajectories (top {top_n} by 2024 rate)",
                      xaxis_title="Year", yaxis_title="Zero-dose rate (%)",
                      yaxis=dict(range=[0, 100]))
    return style_fig(fig, height=560)


# --------------------------------------------------------------------------------------
# D5 forest plot (2026 mean + 95% CI, observed 2024 diamond)
# --------------------------------------------------------------------------------------
def forest_fig(res: pd.DataFrame) -> go.Figure:
    d = res.sort_values("zd_pred_2026_mean").reset_index(drop=True)
    colors = [C.TIER_COLORS.get(str(t), C.STEEL) for t in d["priority_tier"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["zd_pred_2026_mean"], y=d["state"], orientation="h", marker=dict(color=colors),
        error_x=dict(type="data", symmetric=False,
                     array=d["zd_pred_2026_hi95"] - d["zd_pred_2026_mean"],
                     arrayminus=d["zd_pred_2026_mean"] - d["zd_pred_2026_lo95"],
                     color="#222", thickness=1.0, width=3),
        name="2026 predicted", hovertemplate="%{y}<br>2026: %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=d["zd_obs_2024"], y=d["state"], mode="markers", name="2024 observed",
        marker=dict(symbol="diamond", size=8, color="#111", line=dict(color="white", width=0.6)),
        hovertemplate="%{y}<br>2024 observed: %{x:.1f}%<extra></extra>"))
    fig.update_layout(title="Predicted state zero-dose rate, 2026 (95% credible interval)",
                      xaxis_title="Zero-dose rate (%)", yaxis_title="",
                      legend=dict(orientation="h", y=-0.08), xaxis=dict(range=[0, 100]))
    return style_fig(fig, height=820)


# --------------------------------------------------------------------------------------
# D5 state burden bars
# --------------------------------------------------------------------------------------
def burden_bars_fig(res: pd.DataFrame) -> go.Figure:
    d = res.dropna(subset=["zd_count_2026"]).sort_values("zd_count_2026").reset_index(drop=True)
    colors = [C.TIER_COLORS.get(str(t), C.STEEL) for t in d["priority_tier"]]
    fig = go.Figure(go.Bar(
        x=d["zd_count_2026"] / 1e3, y=d["state"], orientation="h", marker=dict(color=colors),
        text=[f"{v/1e3:.0f}k" for v in d["zd_count_2026"]], textposition="outside",
        hovertemplate="%{y}<br>%{x:.0f}k children<extra></extra>"))
    fig.update_layout(title="Estimated zero-dose children by state, 2026 (thousands)",
                      xaxis_title="Zero-dose children (thousands)", yaxis_title="")
    return style_fig(fig, height=820)


# --------------------------------------------------------------------------------------
# D5 zone summary
# --------------------------------------------------------------------------------------
def zone_summary_fig(res: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    zorder = [z for z in C.ZONE_ORDER if z in set(res["zone"])]
    rate = res.groupby("zone")[["zd_obs_2024", "zd_pred_2026_mean", "zd_pred_2027_mean",
                                "zd_pred_2028_mean"]].mean().reindex(zorder)
    burden = res.groupby("zone")["zd_count_2026"].sum().reindex(zorder) / 1e3
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Mean zero-dose rate by zone", "Zero-dose burden by zone, 2026"))
    yr_map = {"zd_obs_2024": 2024, "zd_pred_2026_mean": 2026,
              "zd_pred_2027_mean": 2027, "zd_pred_2028_mean": 2028}
    for col, yr in yr_map.items():
        fig.add_trace(go.Bar(name=str(yr), x=zorder, y=rate[col], marker_color=C.YEAR_COLORS[yr],
                             text=[f"{v:.2f}" for v in rate[col]], textposition="outside",
                             textangle=-90, textfont=dict(size=11), cliponaxis=False), row=1, col=1)
    fig.add_trace(go.Bar(x=zorder, y=burden, marker_color=C.NAVY, showlegend=False,
                         text=[f"{v:.0f}k" for v in burden], textposition="outside",
                         textfont=dict(size=12), cliponaxis=False), row=1, col=2)
    fig.update_yaxes(title_text="Zero-dose rate (%)", range=[0, float(rate.max().max()) * 1.22], row=1, col=1)
    fig.update_yaxes(title_text="Children (thousands)", row=1, col=2)
    fig.update_xaxes(tickfont=dict(size=13))
    fig.update_layout(title="Geopolitical zone summary", barmode="group", font=dict(size=14),
                      legend=dict(orientation="h", y=-0.12, font=dict(size=13)))
    fig.update_annotations(font_size=16)  # subplot titles
    return style_fig(fig, height=520)


# --------------------------------------------------------------------------------------
# D5 Pareto
# --------------------------------------------------------------------------------------
def pareto_fig(pareto: pd.DataFrame, top20_pct: float, n80: int, total: int) -> go.Figure:
    d = pareto.copy()
    x = d["Burden rank"].values
    counts = d["Zero-dose children (est)"].values
    cum = d["Cumulative % of burden"].values
    n = len(d)
    n20 = int(round(0.20 * n))
    bar_colors = [C.ACCENT if i < n20 else C.STEEL for i in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=counts, marker_color=bar_colors, name="Per-LGA burden",
                         hovertemplate="Rank %{x}<br>%{y:,.0f} children<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=cum, yaxis="y2", mode="lines", name="Cumulative %",
                             line=dict(color=C.NAVY, width=2.4),
                             hovertemplate="Rank %{x}<br>%{y:.0f}% cumulative<extra></extra>"))
    fig.add_vline(x=n20, line=dict(color=C.ACCENT, width=1.2, dash="dot"))
    fig.add_annotation(x=n20, y=top20_pct, yref="y2", text=f"Top 20% ({n20} LGAs) = {top20_pct:.0f}%",
                       showarrow=True, arrowhead=2, font=dict(color=C.ACCENT, size=11), ax=60, ay=-30)
    fig.update_layout(
        title=f"LGA Pareto: {total:,.0f} zero-dose children across {n} LGAs",
        xaxis_title="LGAs ranked by estimated zero-dose burden",
        yaxis=dict(title="Zero-dose children per LGA"),
        yaxis2=dict(title="Cumulative % of national burden", overlaying="y", side="right",
                    range=[0, 100], showgrid=False),
        legend=dict(orientation="h", y=-0.18))
    return style_fig(fig, height=480)


# --------------------------------------------------------------------------------------
# Choropleth (continuous rate or categorical hotspot class)
# --------------------------------------------------------------------------------------
def choropleth(gdf, color_col: str, *, categorical: bool, title: str,
               color_map: dict | None = None, colorscale=None,
               range_color=None, legend_title: str = "") -> go.Figure:
    """Built with graph_objects (one trace, one geojson) to stay pandas-3.0 safe."""
    gdf = gdf.reset_index(drop=True).copy()
    gdf["_uid"] = gdf.index.astype(str)
    gj = json.loads(gdf.to_json())
    name_col = "lga" if "lga" in gdf.columns else ("state" if "state" in gdf.columns else "_uid")
    text = gdf[name_col].astype(str).tolist()
    fig = go.Figure()
    if categorical:
        present = [k for k in (color_map or {}) if (gdf[color_col] == k).any()]
        codes = {k: i for i, k in enumerate(present)}
        n = max(len(present), 1)
        z = gdf[color_col].map(codes).fillna(0).tolist()
        cscale = []
        for i, k in enumerate(present):
            cscale.append([i / n, color_map[k]])
            cscale.append([(i + 1) / n, color_map[k]])
        fig.add_trace(go.Choropleth(
            geojson=gj, locations=gdf["_uid"], featureidkey="properties._uid", z=z,
            colorscale=cscale or "Greys", zmin=-0.5, zmax=n - 0.5,
            customdata=gdf[color_col].astype(str), text=text,
            hovertemplate="%{text}<br>%{customdata}<extra></extra>",
            marker_line_color="white", marker_line_width=0.35,
            colorbar=dict(title=clean(legend_title), tickmode="array",
                          tickvals=list(range(len(present))),
                          ticktext=[clean(k) for k in present], len=0.85),
        ))
    else:
        z = pd.to_numeric(gdf[color_col], errors="coerce")
        rc = range_color or [float(z.min()), float(z.max())]
        fig.add_trace(go.Choropleth(
            geojson=gj, locations=gdf["_uid"], featureidkey="properties._uid", z=z,
            colorscale=colorscale or "RdYlGn_r", zmin=rc[0], zmax=rc[1],
            text=text, hovertemplate="%{text}<br>%{z:.1f}<extra></extra>",
            marker_line_color="white", marker_line_width=0.35,
            colorbar=dict(title=clean(legend_title), len=0.85),
        ))
    # Bold state-boundary overlay (dissolve LGAs into states) for clearer admin geography.
    if "state" in gdf.columns and hasattr(gdf, "dissolve"):
        try:
            states = gdf.dissolve(by="state").reset_index()
            states["_sid"] = states.index.astype(str)
            sgj = json.loads(states.to_json())
            fig.add_trace(go.Choropleth(
                geojson=sgj, locations=states["_sid"], featureidkey="properties._sid",
                z=[0] * len(states), colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False, marker_line_color="#0F2233", marker_line_width=1.5,
                hoverinfo="skip"))
        except Exception:
            pass
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="#E8F4F8",
                    showcountries=False, showframe=False)
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=60, b=0))
    return style_fig(fig, height=620)
