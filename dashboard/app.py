import os
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.abspath("."))

from src.models.train_utils import get_trained_fnn
from src.models.tabnet_model import get_trained_tabnet
from src.preprocessing.pipeline import prepare

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable Credit Decisions",
    page_icon="🏦",
    layout="wide"
)

DATASETS = {
    "German Credit": "german_credit",
    "HELOC":         "heloc",
    "Adult Income":  "adult",
    "Give Me Some Credit": "gmsc",
}

# ── Cache model loading ───────────────────────────────────────
@st.cache_resource
def load_fnn(dataset):
    model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
        get_trained_fnn(dataset)
    return model, X_train, X_test, y_train, y_test, X_train_t, X_test_t

@st.cache_resource
def load_tabnet(dataset):
    model, X_train, X_test, y_train, y_test = get_trained_tabnet(dataset)
    return model, X_train, X_test, y_train, y_test

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🏦 XAI Credit Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Model Performance", "🔍 Credit Decision Explorer"]
)

# ══════════════════════════════════════════════════════════════
# PAGE 1: Model Performance
# ══════════════════════════════════════════════════════════════
if page == "📊 Model Performance":
    st.title("📊 Model Performance — FNN vs TabNet")
    st.markdown("Comparison of FNN and TabNet across all four benchmark datasets.")

    # Benchmark tablosu
    bench_path = "reports/fnn_vs_tabnet_benchmark.csv"
    if os.path.exists(bench_path):
        df = pd.read_csv(bench_path)

        st.subheader("AUC-ROC Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(
                df[["dataset", "fnn_auc", "tabnet_auc"]].rename(columns={
                    "dataset": "Dataset",
                    "fnn_auc": "FNN AUC",
                    "tabnet_auc": "TabNet AUC"
                }),
                use_container_width=True
            )

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            x     = np.arange(len(df))
            width = 0.35
            ax.bar(x - width/2, df["fnn_auc"],    width, label="FNN",    color="steelblue")
            ax.bar(x + width/2, df["tabnet_auc"], width, label="TabNet", color="darkorange")
            ax.axhline(y=0.75, color="red", linestyle="--", alpha=0.7,
                       label="Target (0.75)")
            ax.set_xticks(x)
            ax.set_xticklabels(df["dataset"], rotation=15, ha="right")
            ax.set_ylabel("AUC-ROC")
            ax.set_title("FNN vs TabNet AUC-ROC")
            ax.legend()
            ax.set_ylim(0.5, 1.0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Detay tablosu
        st.subheader("Detailed Metrics")
        detail_cols = ["dataset", "fnn_auc", "fnn_recall", "fnn_f1",
                       "tabnet_auc", "tabnet_recall", "tabnet_f1"]
        available = [c for c in detail_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)

        # OpenXAI sonuclari
        xai_path = "reports/openxai_results.csv"
        if os.path.exists(xai_path):
            st.subheader("OpenXAI Evaluation Metrics")
            df_xai = pd.read_csv(xai_path)
            st.dataframe(df_xai, use_container_width=True)

    else:
        st.warning("Benchmark CSV not found. Run `train_all_datasets.py` first.")

# ══════════════════════════════════════════════════════════════
# PAGE 2: Credit Decision Explorer
# ══════════════════════════════════════════════════════════════
else:
    st.title("🔍 Credit Decision Explorer")

    col1, col2 = st.columns([1, 3])

    with col1:
        dataset_name = st.selectbox("Select Dataset", list(DATASETS.keys()))
        dataset      = DATASETS[dataset_name]
        model_choice = st.radio("Model", ["FNN", "TabNet"])
        n_samples    = st.slider("Test sample index", 0, 199, 0)

    dataset_key = dataset

    with col2:
        with st.spinner("Loading model..."):
            if model_choice == "FNN":
                model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
                    load_fnn(dataset_key)
                X_test_t_sample = X_test_t[n_samples:n_samples+1]
                with torch.no_grad():
                    prob = torch.sigmoid(model(X_test_t_sample)).item()
            else:
                model, X_train, X_test, y_train, y_test = \
                    load_tabnet(dataset_key)
                prob = model.predict_proba(
                    X_test.iloc[n_samples:n_samples+1].values)[0][1]

        true_label = y_test.iloc[n_samples]
        decision   = "❌ REJECTED" if prob >= 0.5 else "✅ APPROVED"
        color      = "red" if prob >= 0.5 else "green"

        st.markdown(f"### Decision: :{color}[{decision}]")
        st.metric("Default Probability", f"{prob:.2%}")
        st.metric("True Label", "Default" if true_label == 1 else "Good")

        st.markdown("---")

        # SHAP (sadece FNN)
        if model_choice == "FNN":
            st.subheader("SHAP Feature Attribution")
            with st.spinner("Computing SHAP..."):
                torch.manual_seed(42)
                bg_idx  = torch.randperm(len(X_train_t))[:50]
                bg_data = X_train_t[bg_idx]
                explainer   = shap.GradientExplainer(model, bg_data)
                shap_vals   = explainer.shap_values(X_test_t_sample)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
                if hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
                    shap_vals = shap_vals[:, :, 0]

                feat_names   = X_train.columns.tolist()
                mean_abs     = np.abs(shap_vals[0])
                top10_idx    = np.argsort(mean_abs)[::-1][:10]

                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ["red" if v > 0 else "blue"
                          for v in shap_vals[0][top10_idx[::-1]]]
                ax.barh(
                    [feat_names[i] for i in top10_idx[::-1]],
                    shap_vals[0][top10_idx[::-1]],
                    color=colors
                )
                ax.axvline(x=0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP Value")
                ax.set_title(f"Local SHAP — Sample #{n_samples}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # TabNet attention
        else:
            st.subheader("TabNet Attention Weights")
            with st.spinner("Extracting attention masks..."):
                explain_matrix, _ = model.explain(
                    X_test.iloc[n_samples:n_samples+1].values)
                feat_names = X_train.columns.tolist()
                importance = explain_matrix[0]
                top10_idx  = np.argsort(importance)[::-1][:10]

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(
                    [feat_names[i] for i in top10_idx[::-1]],
                    importance[top10_idx[::-1]],
                    color="darkorange"
                )
                ax.set_xlabel("Attention Weight")
                ax.set_title(f"TabNet Attention — Sample #{n_samples}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Top features tablosu
        st.subheader("Top 10 Important Features")
        if model_choice == "FNN":
            vals = shap_vals[0]
        else:
            vals = importance

        top10 = np.argsort(np.abs(vals))[::-1][:10]
        df_feat = pd.DataFrame({
            "Feature": [feat_names[i] for i in top10],
            "Value":   [f"{vals[i]:.4f}" for i in top10],
            "Direction": ["↑ Risk" if vals[i] > 0 else "↓ Risk"
                          for i in top10]
        })
        st.dataframe(df_feat, use_container_width=True)