# Explainable Credit NN — Explainable Neural Networks for Credit Risk

BAU Capstone Project #1011164 (Spring 2026)

A modular framework that combines neural-network credit risk prediction with
post-hoc explainability (SHAP, Integrated Gradients), counterfactual recourse
(DiCE), and standardized evaluation through the OpenXAI benchmark suite.

## Team

**Computer Engineering:** Oğuz Coşkun, Nil Demirel, Selin Uçansu  
**Management Engineering:** Yaşar Atanur Aksoy, Muhammed Beşir Baybaba, Hulusi Ufuk Güven

**Advisors:** Assist. Prof. M. Aslı Aydın, Assist. Prof. Barış Özcan

## Datasets

- German Credit (UCI / Kaggle)
- HELOC — Home Equity Line of Credit (FICO)
- Adult Income (UCI)
- Give Me Some Credit (Kaggle)

## Repository Structure

    explainable-credit-nn/
    ├── data/                  # raw and processed datasets (gitignored)
    ├── notebooks/             # EDA and exploratory experiments
    ├── src/
    │   ├── preprocessing/     # unified preprocessing pipeline
    │   ├── models/            # FNN, CNN, TabNet implementations
    │   ├── explainers/        # SHAP, IG, DiCE wrappers
    │   └── evaluation/        # OpenXAI metric scripts
    ├── dashboard/             # Streamlit application
    ├── reports/               # generated outputs and figures
    └── requirements.txt

## Setup

```bash
git clone https://github.com/oguzcosskun/explainable-credit-nn.git
cd explainable-credit-nn
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Status

In active development — see status report for current progress.