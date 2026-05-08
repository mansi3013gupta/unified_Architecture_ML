import streamlit as st
import pandas as pd
import sweetviz as sv
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, save_model as cls_save, pull as cls_pull, plot_model as cls_plot
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, save_model as reg_save, pull as reg_pull, plot_model as reg_plot
from pycaret.clustering import setup as clu_setup, create_model as clu_create, plot_model as clu_plot, save_model as clu_save, pull as clu_pull
from pycaret.anomaly import setup as ano_setup, create_model as ano_create, plot_model as ano_plot, save_model as ano_save, pull as ano_pull
from pycaret.time_series import setup as ts_setup, compare_models as ts_compare, save_model as ts_save, pull as ts_pull, plot_model as ts_plot
from pycaret.datasets import get_data
import streamlit.components.v1 as components
import traceback
from ydata_profiling import ProfileReport
import os

import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

def _eda_overview_tab(df):
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", int(df.isnull().sum().sum()))
    c4.metric("Duplicate rows", int(df.duplicated().sum()))
    with st.expander("Summary statistics", expanded=False):
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)


def _eda_missing_tab(df):
    st.subheader("Missing values")
    missing = df.isnull().sum()
    missing_df = missing[missing > 0]
    if len(missing_df) > 0:
        fig = px.bar(
            missing_df,
            labels={"value": "Missing", "index": "Column"},
            title="Missing values per column",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values in this dataset.")


def _eda_distributions_tab(df):
    st.subheader("Distributions")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        st.info("No numeric columns for histograms.")
        return
    column = st.selectbox("Column", numeric_cols, key="eda_hist_col")
    fig = px.histogram(df, x=column, marginal="box", nbins=40, title=f"Distribution — {column}")
    st.plotly_chart(fig, use_container_width=True)
    if len(numeric_cols) >= 2:
        selected = st.multiselect(
            "Scatter matrix columns",
            numeric_cols,
            default=list(numeric_cols[: min(3, len(numeric_cols))]),
            key="eda_scatter_cols",
        )
        if len(selected) >= 2:
            fig2 = px.scatter_matrix(df, dimensions=selected)
            st.plotly_chart(fig2, use_container_width=True)


def _eda_correlation_tab(df):
    st.subheader("Correlation")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        st.info("Need at least two numeric columns for a correlation heatmap.")
        return
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation heatmap")
    st.plotly_chart(fig, use_container_width=True)


def advanced_eda_dashboard(df):
    """Interactive EDA with tabs (Overview, Distributions, Correlation, Missing)."""
    tab_ov, tab_dist, tab_corr, tab_miss = st.tabs(
        ["Overview", "Distributions", "Correlation", "Missing values"]
    )
    with tab_ov:
        _eda_overview_tab(df)
    with tab_dist:
        _eda_distributions_tab(df)
    with tab_corr:
        _eda_correlation_tab(df)
    with tab_miss:
        _eda_missing_tab(df)

def get_all_datasets():
    df = get_data('index')
    return df['Dataset'].to_list()

def show_profile_reports(container):
    if os.path.exists("profile_report.html"):
        with open('profile_report.html', 'r') as f:
            html_content = f.read()
        with container:
            components.html(html_content, height=800, scrolling=True)
    if os.path.exists("sweetviz_report.html"):
        with open('sweetviz_report.html', 'r') as f:
            html_content = f.read()
        with container:
            components.html(html_content, height=800, scrolling=True)

def data_profile(df,container):
    profile = ProfileReport(df, minimal=True)
    profile.to_file("profile_report.html")
    with open('profile_report.html', 'r') as f:
        html_content = f.read()
    with container:
        components.html(html_content, height=800, scrolling=True)
    
import time

def update_progress(progress_bar, step, max_steps):
    progress = int((step / max_steps) * 100)

    text = f"Processing....Step {step}/{max_steps}"
    if step == max_steps:
        text = "Process Completed"

    progress_bar.progress(progress, text=text)

    time.sleep(0.5)   # allow UI update

def display_sweetviz_report(dataframe,container):
    report = sv.analyze(dataframe)
    report.show_html('sweetviz_report.html', open_browser=False)
    with open('sweetviz_report.html', 'r') as f:
        html_content = f.read()
    with container:
        components.html(html_content, height=800, scrolling=True)

def handle_exception(e):
    st.error(
        f"""The app has encountered an error:  
            **{e}**  
            Please check settings - columns selections and model parameters  
            Or
            Create an issue [here](https://github.com/bitbotcoder/mlwiz/issues/new) with the below error details
        """,
        icon="🥺",
    )
    with st.expander("See Error details"):
        st.error(traceback.format_exc())


def store_model_comparison(comparison_df, task):
    try:
        st.session_state["model_comparison_df"] = comparison_df.copy()
    except Exception:
        st.session_state["model_comparison_df"] = comparison_df
    st.session_state["last_task"] = task
    st.session_state["training_complete"] = True


def get_dataset_catalog_df():
    """PyCaret built-in dataset index for the explorer; empty on failure."""
    try:
        return get_data("index")
    except Exception:
        return pd.DataFrame()

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        st.session_state["dataframe"] = df
        st.session_state["dataset_meta"] = {
            "name": uploaded_file.name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing": int(df.isnull().sum().sum()),
            "size_mb": round(uploaded_file.size / (1024 * 1024), 3) if getattr(uploaded_file, "size", None) else None,
        }
        st.success("Dataset loaded successfully.")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        handle_exception(e)

def load_pycaret_dataset(dataset_name):
    try:
        df = get_data(dataset_name)
        st.session_state["dataframe"] = df
        st.session_state["dataset_meta"] = {
            "name": str(dataset_name),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing": int(df.isnull().sum().sum()),
            "size_mb": None,
        }
        st.success(f"Loaded **{dataset_name}**.")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        handle_exception(e)

"""
def eda_report():
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']

        col1,col2 = st.columns([0.6,0.4])
        new_report = col1.toggle(":blue[Generate New]", value=True)
        show_button = col2.button("Show Report")

        pb = st.progress(0, text="Generating Report...")
        cont = st.container(border=False)

        try:
            if show_button:

                if new_report:

                    update_progress(pb,1,4)

                    data_profile(df, cont)

                    update_progress(pb,2,4)

                    display_sweetviz_report(df, cont)

                    update_progress(pb,3,4)

                    show_profile_reports(cont)

                    update_progress(pb,4,4)

                else:
                    show_profile_reports(cont)

        except Exception as e:
            handle_exception(e)

"""

def eda_report():

    if 'dataframe' not in st.session_state:
        st.warning("Upload dataset first")
        return

    df = st.session_state['dataframe']
    st.session_state["eda_completed"] = True

    advanced_eda_dashboard(df)

    st.divider()

    st.subheader("📄 Generate Detailed Report")

    tool = st.selectbox(
        "Select EDA Tool",
        ["None","YData Profiling","Sweetviz"]
    )

    if st.button("Generate Report"):

        if tool == "YData Profiling":

            profile = ProfileReport(df, minimal=True)
            profile.to_file("profile_report.html")

            with open("profile_report.html","r") as f:
                components.html(f.read(), height=800)

        elif tool == "Sweetviz":

            report = sv.analyze(df)
            report.show_html("sweetviz_report.html", open_browser=False)

            with open("sweetviz_report.html","r") as f:
                components.html(f.read(), height=800)


def build_model(task, container):
    
    if 'dataframe' in st.session_state:
        df = st.session_state['dataframe']
        feature_expander = container.expander("Select Columns")

        # Automatically suggest valid options for each type
        numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_candidates = df.select_dtypes(exclude=[np.number]).columns.tolist()

        target_column = feature_expander.selectbox(
            "Select target column",
            df.columns
        ) if task in ["Classification", "Regression", "Time Series Forecasting"] else None

        numerical_columns = feature_expander.multiselect(
            "Select numerical columns",
            numeric_candidates,
            default=numeric_candidates
        )

        categorical_columns = feature_expander.multiselect(
            "Select categorical columns",
            categorical_candidates,
            default=categorical_candidates
        )

        params_expander = container.expander("Tune Parameters")
        # Data Preparation
        handle_missing_data = params_expander.toggle("Handle Missing Data", value=True)
        handle_outliers = params_expander.toggle("Handle Outliers", value=True)
        
        # Scale and Transform
        normalize = params_expander.checkbox("Normalize", value=False)
        normalize_method = params_expander.selectbox("Normalize Method", ["zscore", "minmax", "maxabs", "robust"], index=0 if normalize else -1) if normalize else None
        transformation = params_expander.checkbox("Apply Transformation", value=False)
        transformation_method = params_expander.selectbox("Transformation Method", ["yeo-johnson", "quantile"], index=0 if transformation else -1) if transformation else None
        
        # Feature Engineering
        polynomial_features = params_expander.checkbox("Polynomial Features", value=False)
        polynomial_degree = params_expander.slider("Polynomial Degree", 2, 5, 2) if polynomial_features else None
        
        # Feature Selection
        remove_multicollinearity = params_expander.checkbox("Remove Multicollinearity", value=False)
        multicollinearity_threshold = params_expander.slider("Multicollinearity Threshold", 0.5, 1.0, 0.9) if remove_multicollinearity else None
        
        if not (task == "Anomaly Detection" or task == "Clustering") :
            feature_selection = params_expander.checkbox("Feature Selection", value=False)
            feature_selection_method = params_expander.selectbox("Feature Selection Method", ["classic", "exhaustive"], index=0 if feature_selection else -1) if feature_selection else None
        else:
            feature_selection = None
            feature_selection_method = None
                
        try:
            # Setup arguments for PyCaret
            setup_kwargs = {
                'data': df[numerical_columns + categorical_columns + ([target_column] if target_column else [])],
                'categorical_features': categorical_columns,
                'numeric_features': numerical_columns,
                'target': target_column,
                'preprocess': handle_missing_data,
                'remove_outliers': handle_outliers,
                'normalize': normalize,
                'normalize_method': normalize_method,
                'transformation': transformation,
                'transformation_method': transformation_method,
                'polynomial_features': polynomial_features,
                'polynomial_degree': polynomial_degree,
                'remove_multicollinearity': remove_multicollinearity,
                'multicollinearity_threshold': multicollinearity_threshold,
                'feature_selection': feature_selection,
                'feature_selection_method': feature_selection_method
            }
            pb = st.progress(0, text="Training model...")

            if task == "Classification" and st.button("Run Classification"):
                
                df[target_column] = df[target_column].astype('category')
                
                df.dropna(subset=[target_column] + numerical_columns + categorical_columns, inplace=True)
                
                if len(df) < 2:
                    st.error("Not enough data to split into train and test sets.")
                    return
                update_progress(pb,1,7)
                exp = cls_setup(**setup_kwargs)
                update_progress(pb,2,7)
                best_model = cls_compare()
                update_progress(pb,3,7)
                cmp_df = cls_pull()
                store_model_comparison(cmp_df, task)
                st.dataframe(cmp_df, use_container_width=True)
                update_progress(pb,4,7)
                cls_plot(best_model, plot='auc',display_format="streamlit")
                cls_plot(best_model, plot='confusion_matrix',display_format="streamlit")
                update_progress(pb,5,7)
                st.image(cls_plot(best_model, plot='pr',save=True))
                update_progress(pb,6,7)
                cls_save(best_model, 'best_classification_model')
                st.write('Best Model based on metrics - ')
                st.write(best_model)
                update_progress(pb,7,7)

            elif task == "Regression" and st.button("Run Regression"):
                update_progress(pb,1,7)
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                update_progress(pb,2,7)
                df.dropna(subset=[target_column] + numerical_columns + categorical_columns, inplace=True)
                update_progress(pb,3,7)                
                if len(df) < 2:
                    st.error("Not enough data to split into train and test sets.")
                    return
                
                exp = reg_setup(**setup_kwargs)
                best_model = reg_compare()
                update_progress(pb,4,7)
                cmp_df = reg_pull()
                store_model_comparison(cmp_df, task)
                st.dataframe(cmp_df, use_container_width=True)
                update_progress(pb,5,7)
                st.image(reg_plot(best_model, plot='residuals', save=True))
                st.image(reg_plot(best_model, plot='error', save=True))
                st.image(reg_plot(best_model, plot='error', save=True))
                update_progress(pb,6,7)
                reg_save(best_model, 'best_regression_model')
                st.write('Best Model based on metrics - ')
                st.write(best_model)
                update_progress(pb,7,7)
            elif task == "Clustering" and st.button("Run Clustering"):
                update_progress(pb,1,7)
                df.dropna(subset=numerical_columns + categorical_columns, inplace=True)
                update_progress(pb,2,7)
                setup_kwargs.pop('target')
                setup_kwargs.pop('feature_selection')
                setup_kwargs.pop('feature_selection_method')  
                update_progress(pb,3,7)
                exp = clu_setup(**setup_kwargs)
                best_model = clu_create('kmeans')
                update_progress(pb,4,7)
                clu_plot(best_model, plot='cluster', display_format='streamlit')
                clu_plot(best_model, plot='elbow', display_format='streamlit')
                update_progress(pb,5,7)
                st.write(best_model)
                cmp_df = clu_pull()
                store_model_comparison(cmp_df, task)
                st.dataframe(cmp_df, use_container_width=True)
                update_progress(pb,6,7)
                clu_save(best_model, 'best_clustering_model')
                st.write('Best Model based on metrics - ')
                st.write(best_model)
                update_progress(pb,7,7)

            elif task == "Anomaly Detection" and st.button("Run Anomaly Detection"):
                update_progress(pb,1,7)
                df.dropna(subset=numerical_columns + categorical_columns, inplace=True)
                update_progress(pb,2,7)
                setup_kwargs.pop('target')
                setup_kwargs.pop('feature_selection')
                setup_kwargs.pop('feature_selection_method')        
                update_progress(pb,3,7)
                exp = ano_setup(**setup_kwargs)
                best_model = ano_create('iforest')
                update_progress(pb,4,7)
                ano_plot(best_model, plot='tsne', display_format='streamlit')
                update_progress(pb,5,7)                
                st.write(best_model)
                cmp_df = ano_pull()
                store_model_comparison(cmp_df, task)
                st.dataframe(cmp_df, use_container_width=True)
                update_progress(pb,6,7)
                ano_save(best_model, 'best_anomaly_model')
                st.write('Best Model based on metrics - ')
                st.write(best_model)
                update_progress(pb,7,7)
            elif task == "Time Series Forecasting" :
                date_column = feature_expander.selectbox("Select date column", df.columns)
                if st.button("Run Time Series Forecasting"):
                    update_progress(pb,1,5)
                    df[date_column] = pd.to_datetime(df[date_column])
                    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                    df.dropna(subset=[target_column], inplace=True)
                    update_progress(pb,2,5)                
                    df = df.set_index(date_column).asfreq('D')
                    exp = ts_setup(df, target=target_column, numeric_imputation_target='mean', numeric_imputation_exogenous='mean')
                    best_model = ts_compare()
                    update_progress(pb,3,5)
                    cmp_df = ts_pull()
                    store_model_comparison(cmp_df, task)
                    st.dataframe(cmp_df, use_container_width=True)
                    ts_plot(best_model, plot='forecast', display_format="streamlit")
                    ts_save(best_model, 'best_timeseries_model')
                    update_progress(pb,4,5)
                    st.write('Best Model based on metrics - ')
                    st.write(best_model)
                    update_progress(pb,5,5)
        except Exception as e:
            handle_exception(e)

def download_model(task=None):
    if task is None:
        task = st.session_state.get("last_task", "Classification")
    model_file = None
    if task == "Classification":
        model_file = 'best_classification_model.pkl'
    elif task == "Regression":
        model_file = 'best_regression_model.pkl'
    elif task == "Clustering":
        model_file = 'best_clustering_model.pkl'
    elif task == "Anomaly Detection":
        model_file = 'best_anomaly_model.pkl'
    elif task == "Time Series Forecasting":
        model_file = 'best_timeseries_model.pkl'
    
    if model_file:
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    st.download_button('Download Model', f, file_name=model_file)
            except Exception as e:
                handle_exception(e)
        else:
            st.error("❗No File Found | First Build A ML Model ")
