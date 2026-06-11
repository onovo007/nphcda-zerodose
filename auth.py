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
import os

import streamlit as st

import config as C
from theme import clean, img_data_uri

_LOG_PATH = "/tmp/nphcda_usage_log.csv"


def _secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


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

    logo = img_data_uri(C.LOGO_PATH)
    logo_html = f"<img src='{logo}' style='height:64px'/>" if logo else "<b>NPHCDA</b>"
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown(
            f"<div style='text-align:center;margin-top:6vh'>{logo_html}"
            f"<h2 style='margin:.4rem 0 0'>Zero-Dose Predictive Modelling Platform</h2>"
            f"<p style='color:#6B7A88'>Please identify yourself to continue. Your details are recorded "
            f"for NPHCDA Digital Innovation Hub usage tracking.</p></div>", unsafe_allow_html=True)
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
        st.caption(clean("Tip for the DIH: set ACCESS_CODE and LOG_WEBHOOK in the Space secrets to "
                         "gate access and persist a cross-user login log."))
    return False
