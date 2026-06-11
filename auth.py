"""
Lightweight login + usage tracking for the NPHCDA Digital Innovation Hub.

Behaviour (all configured via Hugging Face Space *Secrets*, never in code):
- REQUIRE_LOGIN (default "true"): if "false", the app is open (guest) - useful for an open demo.
- ACCESS_CODE (optional): if set, users must also enter this shared code to enter.
- LOG_WEBHOOK (optional): a URL (e.g. a Google Apps Script / Zapier endpoint). Each login is
  POSTed there as JSON so the DIH gets a persistent, cross-user record (HF disk is ephemeral).

Every login is also appended to /tmp (ephemeral) and kept in session for an in-app view.
"""
from __future__ import annotations

import csv
import datetime
import json
import os

import streamlit as st

import config as C
from theme import clean, img_data_uri

_LOG_PATH = "/tmp/nphcda_usage_log.csv"


def _secret(key: str, default=None):
    # Hugging Face Space secrets are injected as environment variables, so check those first;
    # fall back to a local .streamlit/secrets.toml for local runs.
    if key in os.environ:
        return os.environ[key]
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _users() -> dict:
    """Optional per-user credentials from a USERS secret: JSON {"user":"pass"} or 'u1:p1,u2:p2'."""
    raw = _secret("USERS")
    if not raw:
        return {}
    try:
        return dict(json.loads(raw))
    except Exception:
        out = {}
        for pair in str(raw).split(","):
            if ":" in pair:
                u, p = pair.split(":", 1)
                out[u.strip()] = p.strip()
        return out


def log_event(rec: dict) -> None:
    rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), **rec}
    st.session_state.setdefault("usage_log", []).append(rec)
    try:
        new = not os.path.exists(_LOG_PATH)
        with open(_LOG_PATH, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rec.keys()))
            if new:
                w.writeheader()
            w.writerow(rec)
    except Exception:
        pass
    url = _secret("LOG_WEBHOOK")
    if url:
        try:
            import requests
            requests.post(url, json=rec, timeout=5)
        except Exception:
            pass


def current_user() -> dict | None:
    return st.session_state.get("auth_user")


def logout() -> None:
    st.session_state.pop("auth_user", None)


def require_login() -> bool:
    """Return True if the user may proceed; otherwise render the login screen and return False."""
    if st.session_state.get("auth_user"):
        return True
    if str(_secret("REQUIRE_LOGIN", "true")).lower() in ("false", "0", "no", "off"):
        st.session_state["auth_user"] = {"name": "Guest", "email": "", "org": ""}
        return True

    # Banner image (falls back to the navy/green gradient).
    banner = img_data_uri(C.LOGIN_BANNER_PATH)
    bg = (f"linear-gradient(115deg, rgba(11,28,42,0.86), rgba(15,82,43,0.62)), url('{banner}')"
          if banner else "linear-gradient(120deg,#10283c,#1F3B57 55%,#1C7A3D 140%)")
    logo = img_data_uri(C.LOGO_PATH)
    logo_html = f"<img src='{logo}' style='height:54px;margin-bottom:8px'/>" if logo else ""
    st.markdown(
        f"<div style=\"background-image:{bg};background-size:cover;background-position:center;"
        f"border-radius:18px;padding:42px 36px;text-align:center;color:#fff;margin-bottom:14px;"
        f"box-shadow:0 16px 40px rgba(10,30,45,.3)\">{logo_html}"
        f"<h1 style='color:#fff;margin:0;font-size:2rem'>Zero-Dose Predictive Modelling Platform</h1>"
        f"<p style='color:rgba(255,255,255,.92);margin:6px 0 0'>NPHCDA - GAVI - UNICEF</p></div>",
        unsafe_allow_html=True)

    users = _users()
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        if users:
            st.markdown("##### Sign in")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                ok = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if ok:
                if users.get(username.strip()) and password == users.get(username.strip()):
                    user = {"name": username.strip(), "email": username.strip(), "org": "NPHCDA"}
                    st.session_state["auth_user"] = user
                    log_event({"event": "login", **user})
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        else:
            st.markdown("##### Identify yourself to continue")
            st.caption(clean("Your details are recorded for NPHCDA Digital Innovation Hub usage tracking."))
            code_required = bool(_secret("ACCESS_CODE"))
            with st.form("login_form"):
                name = st.text_input("Full name")
                email = st.text_input("Email")
                org = st.text_input("Organisation", value="NPHCDA")
                code = st.text_input("Access code", type="password") if code_required else ""
                ok = st.form_submit_button("Enter platform", type="primary", use_container_width=True)
            if ok:
                if not name.strip() or not email.strip():
                    st.error("Please enter your name and email.")
                    return False
                if code_required and code != _secret("ACCESS_CODE"):
                    st.error("Invalid access code.")
                    return False
                user = {"name": name.strip(), "email": email.strip(), "org": org.strip()}
                st.session_state["auth_user"] = user
                log_event({"event": "login", **user})
                st.rerun()
    return False
