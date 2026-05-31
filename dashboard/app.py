import os
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath("."))

from src.models.train_utils import get_trained_fnn
from src.models.tabnet_model import get_trained_tabnet
from src.preprocessing.pipeline import prepare
from src.explainers.dice_explainer import _get_dataset_config, _align_columns, decode, UCI_LABELS

def decode_feature(feat):
    """checking_account_A11 → checking_account: < 0 DM"""
    parts = feat.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in UCI_LABELS:
        return f"{parts[0]}: {UCI_LABELS[parts[1]]}"
    return feat

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

DATASET_SIZES = {
    "german_credit": 199,
    "heloc":         2091,
    "adult":         6512,
    "gmsc":          29999,
}

@st.cache_resource
def load_fnn(dataset):
    model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
        get_trained_fnn(dataset)
    return model, X_train, X_test, y_train, y_test, X_train_t, X_test_t

@st.cache_resource
def load_tabnet(dataset):
    model, X_train, X_test, y_train, y_test = get_trained_tabnet(dataset)
    return model, X_train, X_test, y_train, y_test

st.sidebar.title("🏦 XAI Credit Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Model Performance", "🔍 Credit Decision Explorer"]
)

if page == "📊 Model Performance":
    st.title("📊 Model Performance — FNN vs TabNet")
    st.markdown("Comparison of FNN and TabNet across all four benchmark datasets.")

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

        st.subheader("Detailed Metrics")
        detail_cols = ["dataset", "fnn_auc", "fnn_recall", "fnn_f1",
                       "tabnet_auc", "tabnet_recall", "tabnet_f1"]
        available = [c for c in detail_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)

        xai_path = "reports/openxai_results.csv"
        if os.path.exists(xai_path):
            st.subheader("OpenXAI Evaluation Metrics")
            df_xai = pd.read_csv(xai_path)
            st.dataframe(df_xai, use_container_width=True)

        fairness_path = "reports/fairness_audit.csv"
        if os.path.exists(fairness_path):
            st.subheader("Fairness Audit — Demographic Group Analysis")
            df_fair = pd.read_csv(fairness_path)
            st.dataframe(df_fair, use_container_width=True)
            st.caption(
                "RIS and DiCE Proximity computed separately for demographic "
                "subgroups. Similar values across groups indicate fair behavior."
            )

    else:
        st.warning("Benchmark CSV not found. Run `train_all_datasets.py` first.")

else:
    st.title("🔍 Credit Decision Explorer")

    col1, col2 = st.columns([1, 3])

    with col1:
        dataset_name = st.selectbox("Select Dataset", list(DATASETS.keys()))
        dataset      = DATASETS[dataset_name]
        model_choice = st.radio("Model", ["FNN", "TabNet"])
        max_idx      = DATASET_SIZES.get(dataset, 199)
        n_samples    = st.slider("Test sample index", 0, max_idx, 0)

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

        if model_choice == "FNN":
            st.subheader("SHAP Feature Attribution")
            with st.spinner("Computing SHAP..."):
                torch.manual_seed(42)
                bg_idx  = torch.randperm(len(X_train_t))[:50]
                bg_data = X_train_t[bg_idx]
                explainer = shap.GradientExplainer(model, bg_data)
                shap_vals = explainer.shap_values(X_test_t_sample)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
                if hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
                    shap_vals = shap_vals[:, :, 0]

                feat_names = X_train.columns.tolist()
                if dataset_key == "german_credit":
                    feat_names = [decode_feature(f) for f in feat_names]
                mean_abs   = np.abs(shap_vals[0])
                top10_idx  = np.argsort(mean_abs)[::-1][:10]

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
        else:
            st.subheader("TabNet Attention Weights")
            with st.spinner("Extracting attention masks..."):
                explain_matrix, _ = model.explain(
                    X_test.iloc[n_samples:n_samples+1].values)
                feat_names = X_train.columns.tolist()
                if dataset_key == "german_credit":
                    feat_names = [decode_feature(f) for f in feat_names]
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

        st.subheader("Top 10 Important Features")
        vals = shap_vals[0] if model_choice == "FNN" else importance

        top10 = np.argsort(np.abs(vals))[::-1][:10]
        df_feat = pd.DataFrame({
            "Feature":   [feat_names[i] for i in top10],
            "Value":     [f"{vals[i]:.4f}" for i in top10],
            "Direction": ["↑ Risk" if vals[i] > 0 else "↓ Risk" for i in top10]
        })
        st.dataframe(df_feat, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Action Plan — What should this applicant change?")

        if prob >= 0.5:
            with st.spinner("Generating counterfactual suggestions..."):
                try:
                    import dice_ml
                    from dice_ml import Dice
                    from sklearn.preprocessing import MinMaxScaler

                    df_dice, cat_cols_d, num_cols_d, immutable_d = \
                        _get_dataset_config(dataset_key)

                    scaler_d = MinMaxScaler()
                    df_dice[num_cols_d] = scaler_d.fit_transform(df_dice[num_cols_d])
                    raw_cols = [c for c in df_dice.columns if c != "target"]
                    ref_cols = X_train.columns.tolist()

                    def _wrapper_predict(X_inp):
                        if isinstance(X_inp, pd.DataFrame):
                            X_inp = X_inp[raw_cols]
                        else:
                            X_inp = pd.DataFrame(X_inp, columns=raw_cols)
                        if cat_cols_d:
                            X_enc = pd.get_dummies(X_inp, columns=cat_cols_d,
                                                   drop_first=False)
                        else:
                            X_enc = X_inp.copy()
                        X_enc = _align_columns(X_enc, ref_cols)
                        if model_choice == "FNN":
                            X_t = torch.tensor(X_enc.values, dtype=torch.float32)
                            with torch.no_grad():
                                p = torch.sigmoid(model(X_t)).numpy().flatten()
                        else:
                            p = model.predict_proba(X_enc.values)[:, 1]
                        return np.column_stack([1 - p, p])

                    class WrapperModel:
                        def predict_proba(self, X):
                            return _wrapper_predict(X)

                    cont_feats = num_cols_d if num_cols_d else raw_cols
                    d_dice  = dice_ml.Data(dataframe=df_dice,
                                           continuous_features=cont_feats,
                                           outcome_name="target")
                    m_dice  = dice_ml.Model(model=WrapperModel(), backend="sklearn")
                    exp_dice = Dice(d_dice, m_dice, method="random")

                    _, df_test_dice = train_test_split(
                        df_dice, test_size=0.2, random_state=42,
                        stratify=df_dice["target"])
                    bad_dice = df_test_dice[df_test_dice["target"] == 1]\
                        .reset_index(drop=True)

                    query = bad_dice.iloc[[min(n_samples, len(bad_dice)-1)]][raw_cols]
                    vary  = [f for f in raw_cols if f not in immutable_d]

                    cf    = exp_dice.generate_counterfactuals(
                        query, total_CFs=3,
                        desired_class="opposite",
                        features_to_vary=vary)
                    cf_df = cf.cf_examples_list[0].final_cfs_df

                    if cf_df is not None and len(cf_df) > 0:
                        for j, (_, row_cf) in enumerate(cf_df.iterrows()):
                            changes = []
                            for feat in raw_cols:
                                orig_v = query[feat].values[0]
                                new_v  = row_cf[feat]
                                if str(orig_v) != str(new_v):
                                    if feat in num_cols_d:
                                        idx_n      = num_cols_d.index(feat)
                                        d_o        = np.zeros(len(num_cols_d))
                                        d_n        = np.zeros(len(num_cols_d))
                                        d_o[idx_n] = float(orig_v)
                                        d_n[idx_n] = float(new_v)
                                        r_o = scaler_d.inverse_transform([d_o])[0][idx_n]
                                        r_n = scaler_d.inverse_transform([d_n])[0][idx_n]
                                        if abs(r_o - r_n) >= 0.01:
                                            changes.append(
                                                f"**{feat}**: {r_o:.2f} → {r_n:.2f}")
                                    else:
                                        orig_decoded = decode(orig_v) if dataset_key == "german_credit" else str(orig_v)
                                        new_decoded  = decode(new_v)  if dataset_key == "german_credit" else str(new_v)
                                        changes.append(
                                            f"**{feat}**: {orig_decoded} → {new_decoded}")
                            if changes:
                                st.markdown(f"**Scenario {j+1}:** " + " | ".join(changes))
                    else:
                        st.info("No counterfactual found for this applicant.")

                except Exception as e:
                    st.warning(f"DiCE could not generate suggestions: {e}")
        else:
            st.success("This applicant is already APPROVED — no changes needed.")