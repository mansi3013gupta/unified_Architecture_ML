"""
OpenML-inspired theme: CSS variables, injectors, and small UI helpers for Streamlit.
"""
from __future__ import annotations

import streamlit as st

# Design tokens (light)
COLORS_LIGHT = {
    "primary": "#2563EB",
    "secondary": "#4F46E5",
    "bg": "#F9FAFB",
    "surface": "#FFFFFF",
    "text": "#0F172A",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "success": "#22C55E",
    "error": "#EF4444",
}

# Design tokens (dark)
COLORS_DARK = {
    "primary": "#3B82F6",
    "secondary": "#818CF8",
    "bg": "#0F172A",
    "surface": "#1E293B",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "border": "#334155",
    "success": "#22C55E",
    "error": "#F87171",
}


def _css_block(theme: str) -> str:
    c = COLORS_DARK if theme == "dark" else COLORS_LIGHT
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
  font-family: 'Inter', system-ui, sans-serif !important;
}}

[data-testid="stAppViewContainer"] {{
  background: {c["bg"]} !important;
  color: {c["text"]};
}}

[data-testid="stHeader"] {{
  background: {c["bg"]} !important;
}}

.block-container {{
  padding-top: 1.25rem !important;
  max-width: 1400px !important;
}}

/* Sidebar polish */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, {c["surface"]} 0%, {c["bg"]} 100%) !important;
  border-right: 1px solid {c["border"]} !important;
}}

/* Cards */
.mlwiz-card {{
  background: {c["surface"]};
  border: 1px solid {c["border"]};
  border-radius: 12px;
  padding: 1rem 1.25rem;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
  transition: box-shadow 0.2s ease, transform 0.15s ease;
}}
.mlwiz-card:hover {{
  box-shadow: 0 8px 24px rgba(37, 99, 235, 0.12);
  transform: translateY(-1px);
}}

.mlwiz-metric {{
  background: {c["surface"]};
  border: 1px solid {c["border"]};
  border-radius: 10px;
  padding: 0.85rem 1rem;
}}

/* Top bar */
.mlwiz-topbar {{
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
  padding: 0.5rem 0 1rem 0;
  border-bottom: 1px solid {c["border"]};
  margin-bottom: 1rem;
}}

/* Pipeline */
.mlwiz-pipeline {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  align-items: center;
  margin: 0.5rem 0 1rem 0;
}}
.mlwiz-step {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.35rem 0.65rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 500;
  border: 1px solid {c["border"]};
  background: {c["surface"]};
  color: {c["muted"]};
}}
.mlwiz-step.done {{
  border-color: {c["success"]};
  color: {c["success"]};
  background: rgba(34, 197, 94, 0.08);
}}
.mlwiz-step.active {{
  border-color: {c["primary"]};
  color: {c["primary"]};
  background: rgba(37, 99, 235, 0.08);
}}
.mlwiz-step.locked {{
  opacity: 0.55;
}}

/* Nav buttons in sidebar */
div[data-testid="stSidebar"] .stButton > button {{
  border-radius: 8px !important;
  font-weight: 500 !important;
  transition: background 0.15s ease, transform 0.1s ease !important;
}}
div[data-testid="stSidebar"] .stButton > button:hover {{
  transform: translateX(2px);
}}

/* Primary actions */
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, {c["primary"]} 0%, {c["secondary"]} 100%) !important;
  border: none !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
  gap: 4px;
  background: {c["surface"]};
  padding: 4px;
  border-radius: 10px;
  border: 1px solid {c["border"]};
}}
.stTabs [aria-selected="true"] {{
  background: rgba(37, 99, 235, 0.12) !important;
  color: {c["primary"]} !important;
  border-radius: 8px !important;
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid {c["border"]};
}}

/* Skeleton */
.mlwiz-skeleton {{
  background: linear-gradient(90deg, {c["border"]} 25%, {c["surface"]} 50%, {c["border"]} 75%);
  background-size: 200% 100%;
  animation: mlwiz-shimmer 1.2s infinite;
  border-radius: 8px;
  height: 14px;
  margin: 6px 0;
}}
@keyframes mlwiz-shimmer {{
  0% {{ background-position: 200% 0; }}
  100% {{ background-position: -200% 0; }}
}}

/* Empty / error callouts */
.mlwiz-empty {{
  text-align: center;
  padding: 2.5rem 1.5rem;
  border: 1px dashed {c["border"]};
  border-radius: 12px;
  background: {c["surface"]};
  color: {c["muted"]};
}}
.mlwiz-error-hint {{
  border-left: 4px solid {c["error"]};
  padding: 0.75rem 1rem;
  background: rgba(239, 68, 68, 0.08);
  border-radius: 0 8px 8px 0;
  margin: 0.5rem 0;
}}

/* Hide default Streamlit footer branding space slightly tighter */
footer {{ visibility: hidden; height: 0; }}

/* —— Dashboard: OpenML-style hero + feature strip (fixed palette) —— */
.mlwiz-dash-wrap {{
  margin: 0 0 1.5rem 0;
}}
.mlwiz-hero-openml {{
  background: linear-gradient(105deg, #2563eb 0%, #7c3aed 42%, #ea580c 100%);
  border-radius: 16px 16px 0 0;
  padding: 2.75rem 2.25rem;
  color: #fff;
  box-shadow: 0 12px 40px rgba(37, 99, 235, 0.22);
}}
.mlwiz-hero-openml .mlwiz-hero-grid {{
  display: grid;
  grid-template-columns: 1.05fr 0.95fr;
  gap: 2rem;
  align-items: center;
  max-width: 1120px;
  margin: 0 auto;
}}
@media (max-width: 900px) {{
  .mlwiz-hero-openml .mlwiz-hero-grid {{ grid-template-columns: 1fr; }}
}}
.mlwiz-hero-openml h1 {{
  font-size: clamp(2rem, 4.2vw, 2.85rem);
  font-weight: 700;
  margin: 0 0 0.45rem 0;
  color: #fff !important;
  letter-spacing: -0.03em;
  line-height: 1.1;
}}
.mlwiz-hero-openml .mlwiz-hero-sub {{
  font-size: 1.2rem;
  font-weight: 500;
  opacity: 0.96;
  margin: 0 0 1rem 0;
  color: #fff !important;
}}
.mlwiz-hero-openml .mlwiz-hero-desc {{
  font-size: 1rem;
  line-height: 1.65;
  opacity: 0.94;
  margin: 0;
  color: rgba(255,255,255,0.95) !important;
}}
.mlwiz-hero-openml .mlwiz-hero-art {{
  display: flex;
  justify-content: center;
  align-items: center;
}}
.mlwiz-hero-openml .mlwiz-hero-art svg {{
  width: 100%;
  max-width: 400px;
  height: auto;
  display: block;
  filter: drop-shadow(0 6px 24px rgba(0,0,0,0.18));
}}
.mlwiz-features-openml {{
  background: #ffffff;
  border-radius: 0 0 16px 16px;
  padding: 2.5rem 2rem 2.25rem;
  border: 1px solid #e2e8f0;
  border-top: none;
  box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
}}
.mlwiz-features-openml .mlwiz-feat-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2rem;
  max-width: 1120px;
  margin: 0 auto;
}}
@media (max-width: 768px) {{
  .mlwiz-features-openml .mlwiz-feat-grid {{ grid-template-columns: 1fr; }}
}}
.mlwiz-features-openml .mlwiz-feat-item {{
  text-align: center;
}}
.mlwiz-features-openml .mlwiz-feat-icon {{
  width: 56px;
  height: 56px;
  margin: 0 auto 1rem;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.75rem;
}}
.mlwiz-features-openml .mlwiz-feat-icon.green {{ background: rgba(34, 197, 94, 0.15); }}
.mlwiz-features-openml .mlwiz-feat-icon.blue {{ background: rgba(37, 99, 235, 0.12); }}
.mlwiz-features-openml .mlwiz-feat-icon.red {{ background: rgba(239, 68, 68, 0.12); }}
.mlwiz-features-openml h3 {{
  font-size: 1.05rem;
  font-weight: 700;
  color: #1e293b !important;
  margin: 0 0 0.5rem 0;
}}
.mlwiz-features-openml p {{
  font-size: 0.92rem;
  line-height: 1.55;
  color: #475569 !important;
  margin: 0;
}}
</style>
"""


def inject_theme() -> None:
    theme = st.session_state.get("theme", "light")
    st.markdown(_css_block(theme), unsafe_allow_html=True)


def toggle_theme() -> None:
    cur = st.session_state.get("theme", "light")
    st.session_state.theme = "dark" if cur == "light" else "light"


def card(html_inner: str, class_name: str = "mlwiz-card") -> None:
    st.markdown(f'<div class="{class_name}">{html_inner}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="mlwiz-empty">
  <h3 style="margin:0 0 0.5rem 0; color: inherit;">{title}</h3>
  <p style="margin:0;">{body}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def friendly_error(message: str, suggestion: str | None = None) -> None:
    hint = f"<p style='margin:0.35rem 0 0 0;'>{suggestion}</p>" if suggestion else ""
    st.markdown(
        f'<div class="mlwiz-error-hint"><strong>{message}</strong>{hint}</div>',
        unsafe_allow_html=True,
    )


def skeleton_lines(n: int = 4) -> None:
    parts = "".join('<div class="mlwiz-skeleton"></div>' for _ in range(n))
    st.markdown(parts, unsafe_allow_html=True)


def pipeline_steps(
    labels: list[str],
    current_index: int,
    completed_through: int,
) -> None:
    """
    current_index: active step (0-based)
    completed_through: last step index that is fully done (inclusive), or -1
    """
    chips = []
    for i, lab in enumerate(labels):
        if i == current_index:
            cls = "active"
        elif i <= completed_through and i != current_index:
            cls = "done"
        else:
            cls = "locked"
        mark = "✓ " if cls == "done" else ""
        chips.append(f'<span class="mlwiz-step {cls}">{mark}{i + 1}. {lab}</span>')
    st.markdown('<div class="mlwiz-pipeline">' + "".join(chips) + "</div>", unsafe_allow_html=True)
