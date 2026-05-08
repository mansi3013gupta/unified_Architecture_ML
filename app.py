"""
MLForge — OpenML-inspired AutoML UI (Streamlit).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from ml_pipeline import (
    build_model,
    download_model,
    eda_report,
    get_dataset_catalog_df,
    handle_exception,
    load_data,
    load_pycaret_dataset,
)
from theme import empty_state, friendly_error, inject_theme, pipeline_steps, toggle_theme

VERSION = "0.6.0"

MODEL_FILES = {
    "Classification": "best_classification_model.pkl",
    "Regression": "best_regression_model.pkl",
    "Clustering": "best_clustering_model.pkl",
    "Anomaly Detection": "best_anomaly_model.pkl",
    "Time Series Forecasting": "best_timeseries_model.pkl",
}


def _init_state() -> None:
    defaults = {
        "page": "dashboard",
        "theme": "light",
        "dataset_view": "table",
        "results_view": "table",
        "filter_problem": "Any",
        "filter_rows_max": 1000000,
        "filter_features_max": 10000,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _model_file_path() -> str | None:
    task = st.session_state.get("last_task", "Classification")
    name = MODEL_FILES.get(task)
    return name


def _download_ready() -> bool:
    p = _model_file_path()
    return bool(p and os.path.isfile(p))


def _pipeline_progress() -> tuple[int, int]:
    """Returns (current_step_index 0..4, completed_through inclusive -1..4)."""
    completed = -1
    if "dataframe" in st.session_state:
        completed = 0
    if st.session_state.get("eda_completed"):
        completed = max(completed, 1)
    if st.session_state.get("training_complete"):
        completed = max(completed, 2)
    if st.session_state.get("model_comparison_df") is not None:
        try:
            mdf = st.session_state["model_comparison_df"]
            if hasattr(mdf, "empty") and not mdf.empty:
                completed = max(completed, 3)
        except Exception:
            pass
    if _download_ready():
        completed = max(completed, 4)

    page = st.session_state.get("page", "dashboard")
    step_map = {
        "dashboard": 0,
        "datasets": 0,
        "eda": 1,
        "models": 2,
        "results": 3,
    }
    current = step_map.get(page, 0)
    if page == "results" and _download_ready():
        current = 4
    return current, completed


def _catalog_resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Map logical names to actual column names (PyCaret index schema varies)."""
    out: dict[str, str | None] = {
        "name": None,
        "rows": None,
        "features": None,
        "task": None,
        "target": None,
    }
    if df is None or df.empty:
        return out
    lower = {str(c).lower(): c for c in df.columns}
    for key, patterns in [
        ("name", ["dataset", "name"]),
        ("rows", ["# instances", "instances", "rows"]),
        ("features", ["# attributes", "attributes", "features"]),
        ("task", ["default task", "task", "problem"]),
        ("target", ["target variable", "target"]),
    ]:
        for p in patterns:
            if p in lower:
                out[key] = lower[p]
                break
        if out[key] is None:
            for c in df.columns:
                cl = str(c).lower()
                if any(x in cl for x in patterns):
                    out[key] = c
                    break
    if out["name"] is None:
        out["name"] = df.columns[0]
    return out


def _filter_catalog(
    df: pd.DataFrame,
    cols: dict[str, str | None],
    problem: str,
    max_rows: int,
    max_feat: int,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    rcol, fcol, tcol = cols.get("rows"), cols.get("features"), cols.get("task")
    if rcol and rcol in out.columns:
        nums = pd.to_numeric(out[rcol], errors="coerce").fillna(0)
        out = out[nums <= max_rows]
    if fcol and fcol in out.columns:
        nums = pd.to_numeric(out[fcol], errors="coerce").fillna(0)
        out = out[nums <= max_feat]
    if problem != "Any" and tcol and tcol in out.columns:
        s = out[tcol].astype(str).str.lower()
        if problem == "Classification":
            out = out[s.str.contains("class", na=False)]
        elif problem == "Regression":
            out = out[s.str.contains("regress", na=False)]
    return out


def _render_top_bar() -> None:
    st.markdown('<div class="mlwiz-topbar">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("MLForge · AutoML Workbench")
    with c2:
        theme = st.session_state.get("theme", "light")
        label = "Dark mode" if theme == "light" else "Light mode"
        t1, t2 = st.columns(2)
        with t1:
            if st.button(label, use_container_width=True, key="theme_toggle"):
                toggle_theme()
                st.rerun()
        with t2:
            st.caption("Demo user")
    st.markdown("</div>", unsafe_allow_html=True)


def _sidebar() -> None:
    st.markdown(
        f"""
<div style="padding-bottom:0.75rem;">
  <div style="font-size:1.35rem;font-weight:700;color:#2563EB;">MLForge</div>
  <div style="font-size:0.8rem;color:#64748B;">AutoML Workbench · v{VERSION}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    if os.path.isfile("logo.png"):
        st.image("logo.png", width=120)

    nav = [
        ("dashboard", "Dashboard"),
        ("datasets", "Datasets"),
        ("eda", "EDA"),
        ("models", "Models"),
        ("results", "Results"),
    ]
    st.caption("Navigation")
    for key, label in nav:
        active = st.session_state.page == key
        if st.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.page = key
            st.rerun()

    st.divider()
    st.caption("Workflow")
    labels = ["Upload", "Profiling", "Training", "Evaluation", "Download"]
    cur, done = _pipeline_progress()
    pipeline_steps(labels, cur, done)


def _dashboard_openml_hero_html() -> str:
    """OpenML-inspired hero (gradient + line-art SVG) and three value props."""
    svg = """
<svg viewBox="0 0 420 300" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#fff;stop-opacity:0.95"/>
      <stop offset="100%" style="stop-color:#fff;stop-opacity:0.75"/>
    </linearGradient>
  </defs>
  <rect x="40" y="200" width="120" height="70" rx="8" fill="none" stroke="url(#g)" stroke-width="2.2"/>
  <rect x="260" y="190" width="120" height="80" rx="8" fill="none" stroke="url(#g)" stroke-width="2.2"/>
  <path d="M160 120 Q210 60 260 120 Q210 180 160 120" fill="none" stroke="url(#g)" stroke-width="2.5"/>
  <circle cx="210" cy="120" r="22" fill="none" stroke="url(#g)" stroke-width="2.2"/>
  <path d="M198 118 Q210 108 222 118 M198 124 L222 124" stroke="url(#g)" stroke-width="1.8" fill="none"/>
  <path d="M160 120 L100 160 M260 120 L320 160" stroke="url(#g)" stroke-width="1.8" fill="none"/>
  <path d="M100 160 Q80 200 160 235" stroke="url(#g)" stroke-width="1.6" fill="none" opacity="0.85"/>
  <path d="M320 160 Q340 200 260 235" stroke="url(#g)" stroke-width="1.6" fill="none" opacity="0.85"/>
  <rect x="175" y="248" width="70" height="36" rx="6" fill="rgba(255,255,255,0.15)" stroke="url(#g)" stroke-width="1.5"/>
  <text x="185" y="270" fill="white" font-size="11" font-family="Inter, sans-serif" opacity="0.9">data ↔ model</text>
  <circle cx="55" cy="50" r="18" fill="none" stroke="url(#g)" stroke-width="1.8"/>
  <path d="M48 50h14 M55 43v14" stroke="url(#g)" stroke-width="1.5"/>
  <rect x="330" y="38" width="36" height="44" rx="4" fill="none" stroke="url(#g)" stroke-width="1.8"/>
  <path d="M338 50h20 M338 58h20" stroke="url(#g)" stroke-width="1.4"/>
</svg>
"""
    return f"""
<div class="mlwiz-dash-wrap">
  <div class="mlwiz-hero-openml">
    <div class="mlwiz-hero-grid">
      <div>
        <h1>MLForge</h1>
        <p class="mlwiz-hero-sub">A collaborative machine learning lab</p>
        <p class="mlwiz-hero-desc">
          Upload datasets, explore them interactively, train and compare models automatically,
          and export what works — a single workspace for end-to-end AutoML, inspired by how open science platforms bring data and models together.
        </p>
      </div>
      <div class="mlwiz-hero-art">{svg}</div>
    </div>
  </div>
  <div class="mlwiz-features-openml">
    <div class="mlwiz-feat-grid">
      <div class="mlwiz-feat-item">
        <div class="mlwiz-feat-icon green" title="Data">🗄️</div>
        <h3>AI-ready data</h3>
        <p>Ingest CSV or Excel, profile quality, and browse curated sample datasets to go from raw tables to modeling-ready features.</p>
      </div>
      <div class="mlwiz-feat-item">
        <div class="mlwiz-feat-icon blue" title="Integrations">⚙️</div>
        <h3>ML library integrations</h3>
        <p>Lean on PyCaret-powered training for classification, regression, clustering, anomalies, and time series in one consistent flow.</p>
      </div>
      <div class="mlwiz-feat-item">
        <div class="mlwiz-feat-icon red" title="Results">🧪</div>
        <h3>A treasure trove of ML results</h3>
        <p>Compare leaderboards, highlight strong metrics, and download trained artifacts when your run finishes.</p>
      </div>
    </div>
  </div>
</div>
"""


def _page_dashboard() -> None:
    st.markdown(_dashboard_openml_hero_html(), unsafe_allow_html=True)

    has_data = "dataframe" in st.session_state
    st.markdown("##### Workspace status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dataset loaded", "Yes" if has_data else "No")
    with c2:
        st.metric("Profiling", "Done" if st.session_state.get("eda_completed") else "—")
    with c3:
        st.metric("Training", "Done" if st.session_state.get("training_complete") else "—")
    with c4:
        st.metric("Model file", "Ready" if _download_ready() else "—")

    st.markdown("##### Quick steps")
    a, b, c, d, e = st.columns(5)
    if a.button("1 · Upload", use_container_width=True):
        st.session_state.page = "datasets"
        st.rerun()
    if b.button("2 · EDA", use_container_width=True):
        st.session_state.page = "eda"
        st.rerun()
    if c.button("3 · Train", use_container_width=True):
        st.session_state.page = "models"
        st.rerun()
    if d.button("4 · Results", use_container_width=True):
        st.session_state.page = "results"
        st.rerun()
    if e.button("5 · Export", use_container_width=True):
        st.session_state.page = "results"
        st.rerun()

    if not has_data:
        empty_state("No dataset yet", "Upload a CSV or Excel file, or load a sample from the catalog.")
        if st.button("Go to Datasets", type="primary"):
            st.session_state.page = "datasets"
            st.rerun()


def _page_datasets() -> None:
    st.markdown("### Dataset explorer")
    st.caption("Browse built-in samples (PyCaret) or upload your own — table and card views with filters.")

    up = st.expander("Upload dataset", expanded=True)
    with up:
        u1, u2 = st.columns([2, 1])
        with u1:
            uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx"], label_visibility="collapsed")
        with u2:
            if uploaded and st.button("Load upload", type="primary", use_container_width=True):
                load_data(uploaded)

    catalog = get_dataset_catalog_df()
    cols_map = _catalog_resolve_columns(catalog)

    f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
    with f1:
        opts = ["Any", "Classification", "Regression"]
        ix = opts.index(st.session_state.filter_problem) if st.session_state.filter_problem in opts else 0
        st.session_state.filter_problem = st.selectbox("Problem type", opts, index=ix)
    with f2:
        st.session_state.filter_rows_max = st.number_input("Max rows", 100, 1_000_000, 500_000, 10000)
    with f3:
        st.session_state.filter_features_max = st.number_input("Max features", 2, 10000, 500, 10)
    with f4:
        st.session_state.dataset_view = st.radio(
            "View",
            ["table", "cards"],
            horizontal=True,
            format_func=lambda x: "Table" if x == "table" else "Cards",
        )

    filtered = _filter_catalog(
        catalog,
        cols_map,
        st.session_state.filter_problem,
        int(st.session_state.filter_rows_max),
        int(st.session_state.filter_features_max),
    )

    if catalog.empty:
        friendly_error(
            "Could not load the built-in dataset catalog.",
            "Ensure PyCaret is installed. You can still upload files above.",
        )
    else:
        st.caption(f"Showing {len(filtered)} of {len(catalog)} datasets")

    name_col = cols_map.get("name")
    if name_col and not filtered.empty:
        names = filtered[name_col].astype(str).tolist()
        pick = st.selectbox("Load a sample dataset", names)
        if st.button("Load selected dataset", type="primary"):
            load_pycaret_dataset(pick)
    elif not catalog.empty and filtered.empty:
        st.warning("No datasets match your filters — widen row/feature limits or problem type.")

    display_df = filtered

    if st.session_state.dataset_view == "table":
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        nc, rc, fc, tc, tg = (
            cols_map.get("name"),
            cols_map.get("rows"),
            cols_map.get("features"),
            cols_map.get("task"),
            cols_map.get("target"),
        )
        for _, row in display_df.head(24).iterrows():
            t = str(row.get(nc, "")) if nc else ""
            parts = []
            if rc and rc in row.index:
                parts.append(f"Rows: {row[rc]}")
            if fc and fc in row.index:
                parts.append(f"Features: {row[fc]}")
            if tc and tc in row.index:
                parts.append(str(row[tc]))
            if tg and tg in row.index:
                parts.append(f"Target: {row[tg]}")
            card_html = f"<strong>{t}</strong><br/><span style='color:#64748B;font-size:0.85rem;'>{' · '.join(parts)}</span>"
            st.markdown(f'<div class="mlwiz-card">{card_html}</div>', unsafe_allow_html=True)

    if "dataframe" in st.session_state:
        meta = st.session_state.get("dataset_meta") or {}
        st.subheader("Active session dataset")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Name", meta.get("name", "—"))
        m2.metric("Rows", meta.get("rows", "—"))
        m3.metric("Columns", meta.get("columns", "—"))
        m4.metric("Missing", meta.get("missing", "—"))
        m5.metric("Size (MB)", meta.get("size_mb") if meta.get("size_mb") is not None else "—")


def _page_eda() -> None:
    st.markdown("### Exploratory analysis")
    st.caption("Overview, distributions, correlation, and missing values — interactive Plotly charts.")

    if "dataframe" not in st.session_state:
        empty_state("No dataset uploaded yet", "Load data from the Datasets page to unlock profiling.")
        if st.button("Open Datasets", type="primary", key="eda_to_ds"):
            st.session_state.page = "datasets"
            st.rerun()
        return

    eda_report()


def _page_models() -> None:
    st.markdown("### Model training")
    st.caption("Configure features and AutoML — progress is shown while models train.")

    if "dataframe" not in st.session_state:
        empty_state("No dataset uploaded yet", "Upload a dataset before training.")
        if st.button("Open Datasets", type="primary", key="m_to_ds"):
            st.session_state.page = "datasets"
            st.rerun()
        return

    col1, col2 = st.columns([0.35, 0.65])
    task = col1.selectbox(
        "ML task",
        [
            "Classification",
            "Regression",
            "Clustering",
            "Anomaly Detection",
            "Time Series Forecasting",
        ],
    )
    st.session_state["last_task"] = task
    build_model(task, col2)


def _best_metric_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Accuracy",
        "AUC",
        "F1",
        "Recall",
        "Prec.",
        "R2",
        "MAE",
        "RMSE",
        "MAPE",
    ]
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols[0] if num_cols else None


def _page_results() -> None:
    st.markdown("### Results & export")
    st.caption("Compare models like OpenML leaderboards — sortable table, best run highlighted, optional cards.")

    df = st.session_state.get("model_comparison_df")
    if df is None:
        empty_state("No evaluation results yet", "Train a model on the Models page to populate metrics here.")
        if st.button("Go to training", type="primary", key="r_to_m"):
            st.session_state.page = "models"
            st.rerun()
        return

    if hasattr(df, "empty") and df.empty:
        st.warning("Comparison table is empty.")
        return

    view = st.radio("Layout", ["table", "cards"], horizontal=True, format_func=lambda x: x.title(), key="res_view")
    metric = _best_metric_column(df)
    higher_better = metric not in ("MAE", "RMSE", "MAPE") if metric else True

    if view == "table":
        if metric:
            try:
                styled = (
                    df.style.highlight_max(subset=[metric], color="rgba(34,197,94,0.25)")
                    if higher_better
                    else df.style.highlight_min(subset=[metric], color="rgba(34,197,94,0.25)")
                )
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
        st.caption("Tip: click column headers in the table to sort (Streamlit data grid).")
    else:
        model_col = "Model" if "Model" in df.columns else df.columns[0]
        for i in df.index:
            row = df.loc[i]
            name = str(row.get(model_col, i))
            rest = [c for c in df.columns if c != model_col][:8]
            bits = " · ".join(f"{c}: {row[c]}" for c in rest)
            st.markdown(
                f'<div class="mlwiz-card"><strong>{name}</strong><br/>'
                f'<span style="color:#64748B;font-size:0.9rem;">{bits}</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Download trained model")
    download_model(st.session_state.get("last_task"))
    if not _download_ready():
        st.info("After training completes, the export button appears when the pickle file is on disk.")


# --------------------------------------------------
# APP SHELL
# --------------------------------------------------

st.set_page_config(
    page_title="MLForge · AutoML",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_state()
inject_theme()

with st.sidebar:
    _sidebar()

_render_top_bar()

page = st.session_state.page
try:
    if page == "dashboard":
        _page_dashboard()
    elif page == "datasets":
        _page_datasets()
    elif page == "eda":
        _page_eda()
    elif page == "models":
        _page_models()
    elif page == "results":
        _page_results()
    else:
        st.session_state.page = "dashboard"
        st.rerun()
except Exception as e:
    handle_exception(e)
