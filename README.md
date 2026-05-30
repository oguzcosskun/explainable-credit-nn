# Explainable Credit NN — Explainable Neural Networks for Credit Risk

BAU Capstone Project #1011164 (Spring 2026)

A modular framework that combines neural-network credit risk prediction with
post-hoc explainability (SHAP, Integrated Gradients), counterfactual recourse
(DiCE), and standardized evaluation through the OpenXAI benchmark suite.
Includes both FNN and TabNet architectures benchmarked across four public
credit datasets, with a Streamlit dashboard for interactive exploration.

## Team

**Computer Engineering:** Oguz Coskun, Nil Demirel, Selin Ucansu
**Management Engineering:** Yasar Atanur Aksoy, Muhammed Besir Baybaba, Hulusi Ufuk Guven
**Advisors:** Assist. Prof. M. Asli Aydin, Assist. Prof. Baris Ozcan

## Datasets

| Dataset | Source | Samples | Features |
|---|---|---|---|
| German Credit | UCI | 1,000 | 20 (61 after encoding) |
| HELOC | FICO/Kaggle | 10,459 | 23 |
| Adult Income | UCI | 32,561 | 14 (108 after encoding) |
| Give Me Some Credit | Kaggle | 150,000 | 10 |

## Models

- **FNN** — Feedforward Neural Network (3-layer, BatchNorm, Dropout)
- **TabNet** — Attention-based tabular architecture (inherent interpretability)

All models achieve AUC-ROC >= 0.75 on all datasets (FNN) and 3/4 datasets (TabNet).

## XAI Methods

- **SHAP** — GradientExplainer for global and local feature attribution
- **Integrated Gradients** — Captum-based gradient attribution
- **DiCE** — Counterfactual recourse (actionable "what-if" scenarios)
- **OpenXAI** — Faithfulness, Stability (RIS), Consistency evaluation

## Repository Structure

```
explainable-credit-nn/
|-- data/                  # raw datasets (gitignored)
|-- models/                # trained checkpoints (.pt, .zip)
|-- src/
|   |-- preprocessing/     # unified pipeline (split-first, no leakage)
|   |-- models/            # FNN, TabNet, train_utils
|   |-- explainers/        # SHAP, IG, DiCE, TabNet explainers
|   `-- evaluation/        # OpenXAI metrics
|-- dashboard/             # Streamlit application
|-- reports/               # CSVs and figures
|   |-- fnn_vs_tabnet_benchmark.csv
|   |-- openxai_results.csv
|   |-- dice_results.csv
|   `-- figures/           # per-dataset XAI plots
`-- requirements.txt
```

## Setup

```bash
git clone https://github.com/oguzcosskun/explainable-credit-nn.git
cd explainable-credit-nn
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place `heloc_dataset_v1.csv` and `cs-training.csv` in `data/raw/` (Kaggle download required).

## Usage

```bash
# Train all models
python train_all_datasets.py

# Run XAI methods
python src/explainers/shap_explainer.py --dataset all
python src/explainers/ig_explainer.py --dataset all
python src/explainers/dice_explainer.py --dataset all
python src/evaluation/openxai_metrics.py --dataset all

# Launch dashboard
streamlit run dashboard/app.py
```

## Key Results

| Dataset | FNN AUC | TabNet AUC |
|---|---|---|
| German Credit | 0.7724 | 0.7121 |
| HELOC | 0.7933 | 0.7875 |
| Adult Income | 0.9118 | 0.8921 |
| Give Me Some Credit | 0.8334 | 0.8546 |

## Status

Core technical implementation complete. Remaining: usability study (ENM), fairness audit, final report.