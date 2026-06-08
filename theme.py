"""
Visual theme: global CSS, a shared Plotly template, KPI cards, section headers, and a
house-style text sanitiser (no em/en dashes, no section signs). Import and call
inject_theme() once at the top of app.py.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import config as C

# --------------------------------------------------------------------------------------
# House-style sanitiser
# --------------------------------------------------------------------------------------
_BAD = {
    "—": "-", "–": "-", "‒": "-", "−": "-", "§": "",
    "‘": "'", "’": "'", "“": '"', "”": '"',
}


def clean(text) -> str:
    """Strip forbidden glyphs from any user-facing string."""
    s = str(text)
    for bad, good in _BAD.items():
        s = s.replace(bad, good)
    return s


# --------------------------------------------------------------------------------------
# Plotly template
# --------------------------------------------------------------------------------------
def _register_template() -> None:
    t = go.layout.Template()
    t.layout.font = dict(family=C.FONT_BODY, size=13, color=C.INK)
    t.layout.paper_bgcolor = C.PAPER
    t.layout.plot_bgcolor = C.PAPER
    t.layout.colorway = [C.NAVY, C.STEEL, C.ACCENT, C.GOLD, "#1A9850", "#880E4F", C.MUTE]
    t.layout.title = dict(font=dict(family=C.FONT_HEAD, size=18, color=C.NAVY), x=0.0, xanchor="left")
    axis = dict(
        showgrid=True, gridcolor="rgba(120,130,140,0.14)", gridwidth=1,
        zeroline=False, linecolor="rgba(31,59,87,0.45)", linewidth=1,
        ticks="outside", tickcolor="rgba(120,130,140,0.4)",
        title=dict(font=dict(size=13, color=C.MUTE)),
    )
    t.layout.xaxis = axis
    t.layout.yaxis = axis
    t.layout.legend = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(120,130,140,0.25)",
                           borderwidth=1, font=dict(size=11))
    t.layout.margin = dict(l=70, r=40, t=70, b=60)
    t.layout.hoverlabel = dict(font=dict(family=C.FONT_BODY, size=12), bgcolor="white")
    pio.templates["nphcda"] = t
    pio.templates.default = "plotly_white+nphcda"


_register_template()


def style_fig(fig: go.Figure, height: int | None = None, title: str | None = None) -> go.Figure:
    fig.update_layout(template="plotly_white+nphcda")
    if title is not None:
        fig.update_layout(title_text=clean(title))
    if height is not None:
        fig.update_layout(height=height)
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title_text=clean(fig.layout.title.text))
    return fig


# --------------------------------------------------------------------------------------
# Global CSS (premium, cohesive)
# --------------------------------------------------------------------------------------
def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:wght@600;700&display=swap');

        :root {{
            --navy:{C.NAVY}; --steel:{C.STEEL}; --accent:{C.ACCENT}; --gold:{C.GOLD};
            --mute:{C.MUTE}; --ink:{C.INK}; --panel:#FFFFFF; --bg:#EEF3F8;
        }}
        html, body, [class*="css"] {{ font-family:{C.FONT_BODY}; color:var(--ink); }}
        .stApp {{ background:
            radial-gradient(1200px 480px at 18% -8%, #E3ECF5 0%, rgba(227,236,245,0) 60%),
            linear-gradient(180deg, #EEF3F8 0%, #F4F7FB 100%); }}
        .block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1480px; }}
        h1,h2,h3,h4 {{ font-family:{C.FONT_HEAD}; color:var(--navy); letter-spacing:.2px; }}
        a {{ color:var(--steel); }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background:linear-gradient(180deg,#142A3E 0%, #1F3B57 55%, #22597a 140%);
            border-right:1px solid rgba(0,0,0,0.15);
        }}
        section[data-testid="stSidebar"] * {{ color:#E9F1F7 !important; }}
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {{ color:#FFFFFF !important; }}
        section[data-testid="stSidebar"] [data-baseweb="input"] input {{ color:#10212f !important; }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
            padding:8px 12px; border-radius:9px; margin-bottom:3px; transition:background .15s ease;
        }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
            background:rgba(255,255,255,0.08);
        }}

        /* Hero */
        .hero {{
            background:linear-gradient(120deg,#16293c 0%, {C.NAVY} 45%, {C.STEEL} 100%);
            border-radius:20px; padding:34px 40px; margin-bottom:18px; color:#fff;
            box-shadow:0 18px 44px rgba(15,40,60,0.30);
            border:1px solid rgba(255,255,255,0.06); position:relative; overflow:hidden;
        }}
        .hero::after {{ content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px;
            background:radial-gradient(circle, rgba(200,144,42,0.22) 0%, rgba(200,144,42,0) 70%); }}
        .hero h1 {{ color:#fff; font-size:2.15rem; margin:0 0 8px 0; font-weight:800; }}
        .hero p {{ color:rgba(255,255,255,0.92); font-size:1.04rem; margin:0; max-width:60rem; }}
        .hero .pill {{ display:inline-block; background:rgba(255,255,255,0.12);
            border:1px solid rgba(255,255,255,0.30); border-radius:999px; padding:6px 15px;
            margin:12px 8px 0 0; font-size:.82rem; font-weight:600; backdrop-filter:blur(4px); }}

        /* KPI cards */
        .kpi-wrap {{ display:flex; gap:16px; flex-wrap:wrap; margin:10px 0 6px 0; }}
        .kpi {{ flex:1; min-width:170px; background:var(--panel); border-radius:16px; padding:16px 18px;
            border:1px solid rgba(31,59,87,0.08); border-top:4px solid var(--steel);
            box-shadow:0 6px 18px rgba(15,40,60,0.07); transition:transform .15s ease, box-shadow .15s ease; }}
        .kpi:hover {{ transform:translateY(-3px); box-shadow:0 12px 26px rgba(15,40,60,0.12); }}
        .kpi .label {{ color:var(--mute); font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.7px; }}
        .kpi .value {{ color:var(--navy); font-size:1.8rem; font-weight:800; font-family:{C.FONT_HEAD}; line-height:1.1; margin-top:3px; }}
        .kpi .sub {{ color:var(--mute); font-size:.8rem; margin-top:3px; }}

        /* Section headers */
        .section-head {{ border-left:4px solid var(--gold); padding:1px 0 1px 14px; margin:24px 0 8px 0; }}
        .section-head h3 {{ margin:0; font-size:1.18rem; }}
        .section-head p {{ margin:3px 0 0 0; color:var(--mute); font-size:.9rem; }}

        /* Tabs - blended with the background, gold underline on the active tab */
        .stTabs [data-baseweb="tab-list"] {{ gap:4px; background:transparent;
            border-bottom:2px solid rgba(31,59,87,0.10); padding-bottom:0; }}
        .stTabs [data-baseweb="tab"] {{ background:transparent; border:none; color:var(--mute);
            font-weight:600; padding:10px 18px; border-radius:10px 10px 0 0; transition:all .15s ease; }}
        .stTabs [data-baseweb="tab"]:hover {{ color:var(--navy); background:rgba(46,110,142,0.07); }}
        .stTabs [aria-selected="true"] {{ color:var(--navy) !important; background:var(--panel);
            box-shadow:inset 0 3px 0 var(--gold), 0 -1px 8px rgba(15,40,60,0.05); }}
        .stTabs [data-baseweb="tab-highlight"] {{ background:transparent; }}

        /* Buttons */
        .stButton button {{ border-radius:10px; font-weight:600; border:1px solid rgba(31,59,87,0.18); }}
        .stButton button[kind="primary"] {{ background:var(--accent); border:none; color:#fff;
            box-shadow:0 6px 16px rgba(192,57,43,0.25); }}
        .stButton button[kind="primary"]:hover {{ filter:brightness(1.05); transform:translateY(-1px); }}
        .stDownloadButton button {{ border-radius:10px; font-weight:600; }}

        /* Expanders (AI interpretation blocks) */
        .streamlit-expanderHeader, details summary {{ font-weight:600; color:var(--navy); }}
        [data-testid="stExpander"] {{ border:1px solid rgba(31,59,87,0.12); border-radius:12px;
            background:linear-gradient(180deg,#FBFDFF 0%, #F4F8FC 100%); box-shadow:0 3px 10px rgba(15,40,60,0.04); }}

        /* Dataframes + alerts */
        [data-testid="stDataFrame"] {{ border:1px solid rgba(31,59,87,0.10); border-radius:12px; overflow:hidden; }}
        [data-testid="stMetricValue"] {{ color:var(--navy); }}
        .footnote {{ color:var(--mute); font-size:.8rem; font-style:italic; margin-top:6px; }}
        hr {{ border-color:rgba(31,59,87,0.10); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, pills: list[str] | None = None) -> None:
    pill_html = "".join(f"<span class='pill'>{clean(p)}</span>" for p in (pills or []))
    st.markdown(
        f"<div class='hero'><h1>{clean(title)}</h1><p>{clean(subtitle)}</p>{pill_html}</div>",
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    sub = f"<p>{clean(subtitle)}</p>" if subtitle else ""
    st.markdown(f"<div class='section-head'><h3>{clean(title)}</h3>{sub}</div>", unsafe_allow_html=True)


def kpi_row(cards: list[dict]) -> None:
    """cards: list of {label, value, sub?, color?}."""
    html = "<div class='kpi-wrap'>"
    for c in cards:
        color = c.get("color", C.STEEL)
        sub = f"<div class='sub'>{clean(c.get('sub',''))}</div>" if c.get("sub") else ""
        html += (f"<div class='kpi' style='border-top-color:{color}'>"
                 f"<div class='label'>{clean(c['label'])}</div>"
                 f"<div class='value'>{clean(c['value'])}</div>{sub}</div>")
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
