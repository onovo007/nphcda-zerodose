"""
Visual theme: global CSS, a shared Plotly template, KPI cards, section headers, a branded
hero (optionally photographic), a sidebar brand block, and a house-style text sanitiser
(no em/en dashes, no section signs). Import and call inject_theme() once at the top of app.py.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
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
    s = str(text)
    for bad, good in _BAD.items():
        s = s.replace(bad, good)
    return s


def highlight_below(df: pd.DataFrame, col: str, thresh: float = 80.0):
    """Return a Styler that flags values below the threshold in red (for at-risk projections)."""
    if df is None or df.empty or col not in df.columns:
        return df

    def _row(s):
        return ["color:#C0392B; font-weight:700" if (pd.notna(v) and v < thresh) else "" for v in s]

    try:
        return df.style.apply(_row, subset=[col])
    except Exception:
        return df


def img_data_uri(path: Path) -> str | None:
    """Return a base64 data URI for an image on disk, or None if missing."""
    try:
        if not path or not Path(path).exists():
            return None
        mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Plotly template
# --------------------------------------------------------------------------------------
def _register_template() -> None:
    t = go.layout.Template()
    t.layout.font = dict(family=C.FONT_BODY, size=13, color=C.INK)
    t.layout.paper_bgcolor = C.PAPER
    t.layout.plot_bgcolor = C.PAPER
    t.layout.colorway = [C.NAVY, C.STEEL, C.ACCENT, C.GOLD, C.NPHCDA_GREEN, "#880E4F", C.MUTE]
    t.layout.title = dict(font=dict(family=C.FONT_HEAD, size=18, color=C.NAVY), x=0.0, xanchor="left")
    axis = dict(showgrid=True, gridcolor="rgba(120,130,140,0.14)", gridwidth=1, zeroline=False,
                linecolor="rgba(31,59,87,0.45)", linewidth=1, ticks="outside",
                tickcolor="rgba(120,130,140,0.4)", title=dict(font=dict(size=13, color=C.MUTE)))
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
# Global CSS (premium, tech-grade)
# --------------------------------------------------------------------------------------
def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600;700&display=swap');

        :root {{
            --navy:{C.NAVY}; --steel:{C.STEEL}; --accent:{C.ACCENT}; --gold:{C.GOLD};
            --green:{C.NPHCDA_GREEN}; --green-dk:{C.NPHCDA_GREEN_DK};
            --mute:{C.MUTE}; --ink:{C.INK}; --panel:#FFFFFF;
        }}
        html, body, [class*="css"] {{ font-family:{C.FONT_BODY}; color:var(--ink); }}
        .stApp {{
            background:
              radial-gradient(900px 420px at 88% -6%, rgba(28,122,61,0.10) 0%, rgba(28,122,61,0) 60%),
              radial-gradient(1100px 460px at 12% -8%, rgba(31,59,87,0.12) 0%, rgba(31,59,87,0) 60%),
              linear-gradient(180deg, #EAF0F6 0%, #F4F7FB 100%);
        }}
        .block-container {{ padding-top:1.2rem; padding-bottom:3rem; max-width:1500px; }}
        h1,h2,h3,h4 {{ font-family:{C.FONT_HEAD}; color:var(--navy); letter-spacing:.2px; }}
        a {{ color:var(--steel); }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background:linear-gradient(180deg,#0E2233 0%, #163049 50%, #14502B 220%);
            border-right:1px solid rgba(0,0,0,0.18);
        }}
        section[data-testid="stSidebar"] * {{ color:#E9F1F7 !important; }}
        section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {{ color:#fff !important; }}
        section[data-testid="stSidebar"] [data-baseweb="input"] input {{ color:#10212f !important; }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
            padding:9px 12px; border-radius:10px; margin-bottom:3px; transition:all .15s ease;
            border:1px solid transparent;
        }}
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
            background:rgba(255,255,255,0.08); border-color:rgba(255,255,255,0.10);
        }}
        .brand-chip {{
            background:#fff; border-radius:14px; padding:12px 14px; margin:2px 0 6px 0;
            display:flex; align-items:center; gap:10px; box-shadow:0 6px 18px rgba(0,0,0,0.25);
        }}
        .brand-chip img {{ height:42px; width:auto; }}
        .brand-chip .bt {{ font-family:{C.FONT_HEAD}; color:var(--green-dk)!important; font-weight:700;
            font-size:.92rem; line-height:1.05; }}
        .brand-chip .bs {{ color:#5B6B79!important; font-size:.66rem; letter-spacing:.3px; }}

        /* Hero (photographic or gradient) */
        @keyframes heroPan {{ 0%{{background-position:50% 40%}} 100%{{background-position:50% 60%}} }}
        .hero {{
            position:relative; border-radius:22px; padding:40px 44px; margin-bottom:18px; color:#fff;
            box-shadow:0 22px 54px rgba(10,30,45,0.34); border:1px solid rgba(255,255,255,0.08);
            overflow:hidden; background-size:cover; background-position:center;
        }}
        .hero.hero-anim {{ animation:heroPan 18s ease-in-out infinite alternate; }}
        .hero::before {{ content:""; position:absolute; inset:0;
            background:radial-gradient(120% 120% at 80% 10%, rgba(28,122,61,0.30) 0%, rgba(28,122,61,0) 55%); }}
        .hero h1 {{ color:#fff; font-family:{C.FONT_HEAD}; font-size:2.25rem; margin:0 0 8px 0;
            font-weight:700; letter-spacing:.2px; text-shadow:0 3px 18px rgba(0,0,0,0.45); position:relative; }}
        .hero p {{ color:rgba(255,255,255,0.94); font-size:1.05rem; margin:0; max-width:62rem;
            text-shadow:0 2px 12px rgba(0,0,0,0.5); position:relative; }}
        .hero .pill {{ display:inline-block; background:rgba(255,255,255,0.14);
            border:1px solid rgba(255,255,255,0.34); border-radius:999px; padding:6px 15px;
            margin:14px 8px 0 0; font-size:.82rem; font-weight:600; backdrop-filter:blur(6px); position:relative; }}
        .hero .pill.green {{ background:rgba(28,122,61,0.40); border-color:rgba(255,255,255,0.4); }}

        /* Per-domain banner */
        .dbanner {{ position:relative; border-radius:16px; padding:22px 30px; margin-bottom:14px;
            color:#fff; background-size:cover; background-position:center;
            box-shadow:0 12px 32px rgba(10,30,45,0.26); border:1px solid rgba(255,255,255,0.08); }}
        .dbanner h2 {{ color:#fff; font-family:{C.FONT_HEAD}; font-size:1.6rem; margin:0 0 4px 0;
            text-shadow:0 2px 14px rgba(0,0,0,0.5); }}
        .dbanner p {{ color:rgba(255,255,255,0.93); margin:0; font-size:.95rem; max-width:65rem;
            text-shadow:0 2px 10px rgba(0,0,0,0.5); }}

        /* KPI cards (glassy, tech) */
        .kpi-wrap {{ display:flex; gap:16px; flex-wrap:wrap; margin:12px 0 6px 0; }}
        .kpi {{ flex:1; min-width:172px; position:relative; border-radius:16px; padding:16px 18px;
            background:linear-gradient(180deg,#ffffff 0%, #f7fafc 100%);
            border:1px solid rgba(31,59,87,0.09); box-shadow:0 8px 22px rgba(15,40,60,0.08);
            transition:transform .16s ease, box-shadow .16s ease; overflow:hidden; }}
        .kpi::before {{ content:""; position:absolute; left:0; top:0; height:4px; width:100%;
            background:linear-gradient(90deg, var(--accent-c, var(--steel)), transparent 85%); }}
        .kpi:hover {{ transform:translateY(-4px); box-shadow:0 16px 32px rgba(15,40,60,0.14); }}
        .kpi .label {{ color:var(--mute); font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.7px; }}
        .kpi .value {{ color:var(--navy); font-size:1.85rem; font-weight:700; font-family:{C.FONT_BODY};
            font-feature-settings:'tnum' 1; line-height:1.1; margin-top:3px; }}
        .kpi .sub {{ color:var(--mute); font-size:.8rem; margin-top:3px; }}

        /* Section headers */
        .section-head {{ border-left:4px solid var(--green); padding:1px 0 1px 14px; margin:24px 0 8px 0; }}
        .section-head h3 {{ margin:0; font-size:1.18rem; }}
        .section-head p {{ margin:3px 0 0 0; color:var(--mute); font-size:.9rem; }}

        /* Tabs - blended, gold underline on active */
        .stTabs [data-baseweb="tab-list"] {{ gap:4px; background:transparent;
            border-bottom:2px solid rgba(31,59,87,0.10); }}
        .stTabs [data-baseweb="tab"] {{ background:transparent; border:none; color:var(--mute);
            font-weight:600; padding:10px 18px; border-radius:10px 10px 0 0; transition:all .15s ease; }}
        .stTabs [data-baseweb="tab"]:hover {{ color:var(--navy); background:rgba(46,110,142,0.07); }}
        .stTabs [aria-selected="true"] {{ color:var(--navy) !important; background:var(--panel);
            box-shadow:inset 0 3px 0 var(--gold), 0 -1px 8px rgba(15,40,60,0.05); }}
        .stTabs [data-baseweb="tab-highlight"] {{ background:transparent; }}

        /* Buttons */
        .stButton button {{ border-radius:10px; font-weight:600; border:1px solid rgba(31,59,87,0.18); transition:all .15s ease; }}
        .stButton button:hover {{ transform:translateY(-1px); }}
        .stButton button[kind="primary"] {{ background:linear-gradient(135deg, var(--accent), #a93226);
            border:none; color:#fff; box-shadow:0 6px 16px rgba(192,57,43,0.28); }}
        .stDownloadButton button {{ border-radius:10px; font-weight:600; }}

        /* Expanders + chat (AI blocks) */
        [data-testid="stExpander"] {{ border:1px solid rgba(28,122,61,0.18); border-radius:13px;
            background:linear-gradient(180deg,#FBFFFD 0%, #F3F8F4 100%); box-shadow:0 3px 10px rgba(15,40,60,0.04); }}
        [data-testid="stChatMessage"] {{ border-radius:12px; }}

        /* Dataframes + misc */
        [data-testid="stDataFrame"] {{ border:1px solid rgba(31,59,87,0.10); border-radius:12px; overflow:hidden; }}
        [data-testid="stMetricValue"] {{ color:var(--navy); }}
        .footnote {{ color:var(--mute); font-size:.8rem; font-style:italic; margin-top:6px; }}
        hr {{ border-color:rgba(31,59,87,0.10); }}
        #MainMenu, footer {{ visibility:hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    """Branded logo chip at the top of the sidebar (falls back to text)."""
    uri = img_data_uri(C.LOGO_PATH)
    if uri:
        st.markdown(
            f"<div class='brand-chip'><img src='{uri}' alt='NPHCDA'/>"
            f"<div><div class='bt'>Zero-Dose Platform</div>"
            f"<div class='bs'>NPHCDA - GAVI - UNICEF</div></div></div>",
            unsafe_allow_html=True)
    else:
        st.markdown("### NPHCDA Zero-Dose Platform")


def hero(title: str, subtitle: str, pills: list[str] | None = None) -> None:
    uri = img_data_uri(C.HERO_PATH)
    if uri:
        # Photographic hero: navy->green gradient overlay keeps the text legible and on-brand.
        style = (f"background-image:linear-gradient(115deg, rgba(11,28,42,0.90) 0%, "
                 f"rgba(15,40,60,0.78) 42%, rgba(15,82,43,0.62) 100%), url('{uri}');")
        cls = "hero hero-anim"
    else:
        style = ("background:linear-gradient(120deg,#10283c 0%, #1F3B57 45%, #1C7A3D 130%);")
        cls = "hero"
    pill_html = "".join(
        f"<span class='pill{' green' if i % 2 else ''}'>{clean(p)}</span>"
        for i, p in enumerate(pills or []))
    st.markdown(
        f"<div class='{cls}' style=\"{style}\"><h1>{clean(title)}</h1>"
        f"<p>{clean(subtitle)}</p>{pill_html}</div>",
        unsafe_allow_html=True)


def domain_banner(img_name: str, title: str, subtitle: str = "") -> None:
    """A slim photographic banner for a domain page (falls back to a gradient)."""
    uri = img_data_uri(C.ASSETS_DIR / img_name)
    if uri:
        style = (f"background-image:linear-gradient(110deg, rgba(11,28,42,0.92) 0%, "
                 f"rgba(15,40,60,0.72) 45%, rgba(15,82,43,0.55) 100%), url('{uri}');")
    else:
        style = "background:linear-gradient(120deg,#10283c 0%,#1F3B57 55%,#1C7A3D 140%);"
    sub = f"<p>{clean(subtitle)}</p>" if subtitle else ""
    st.markdown(f"<div class='dbanner' style=\"{style}\"><h2>{clean(title)}</h2>{sub}</div>",
                unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    sub = f"<p>{clean(subtitle)}</p>" if subtitle else ""
    st.markdown(f"<div class='section-head'><h3>{clean(title)}</h3>{sub}</div>", unsafe_allow_html=True)


def kpi_row(cards: list[dict]) -> None:
    """cards: list of {label, value, sub?, color?}."""
    html = "<div class='kpi-wrap'>"
    for c in cards:
        color = c.get("color", C.STEEL)
        sub = f"<div class='sub'>{clean(c.get('sub',''))}</div>" if c.get("sub") else ""
        html += (f"<div class='kpi' style='--accent-c:{color}'>"
                 f"<div class='label'>{clean(c['label'])}</div>"
                 f"<div class='value'>{clean(c['value'])}</div>{sub}</div>")
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
