# Explainable and Cross-Dataset Generalizable Machine Learning Framework for Network Intrusion Detection

An MSc research framework demonstrating that high in-dataset accuracy in ML-based
Network Intrusion Detection Systems (NIDS) does not guarantee real-world
generalization — and using SHAP/LIME explainability to understand why.

## Project Summary

This framework trains machine learning models on CICIDS2017 to detect DDoS attacks,
explains their predictions using SHAP and LIME, and rigorously tests whether the
trained model generalizes to **three independent external datasets**: CSE-CIC-IDS2018,
UNSW-NB15, and CICDDoS2019.

**Headline finding:** the final Random Forest achieves 99.99% F1 in-dataset but F1 = 0 (FNR = 1.0)
on all three external datasets at the default threshold. A follow-up comparison of all four models
with an oracle-threshold analysis shows no model beats the trivial "flag everything" baseline on
CICDDoS2019 or UNSW-NB15, while on CSE-CIC-IDS2018 Logistic Regression and XGBoost retain ranking
signal (ROC-AUC 0.986 / 0.960) that the default threshold fails to exploit.

## Research Scope

- **Task:** Binary classification, BENIGN vs DDoS/Attack
- **Source (training) dataset:** CICIDS2017, Friday-Afternoon DDoS capture
- **Target (cross-dataset test) datasets:**
  - CSE-CIC-IDS2018 (Wed-21-02, DDoS-HOIC + DDoS-LOIC-UDP) — same extraction tool, 20/20 features aligned
  - UNSW-NB15 — different extraction tool (Argus/Bro-IDS), only 7/20 features had any conceptual counterpart
  - CICDDoS2019 (UDPLag) — same extraction tool, 20/20 features aligned
- **Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Explainability:** SHAP (global + local) and LIME (local, cross-checked against SHAP)
- **Feature selection:** Correlation filtering (67→42 features) + Random Forest importance ranking (42→20 features)

## Project Structure

```
Network_IDS_Framework/
├── app/                        # Streamlit demo UI (4 pages)
│   ├── app.py
│   └── pages_content/
├── configs/
│   └── config.yaml             # Central config: paths, random seed, dataset scope
├── datasets/                   # Raw downloaded datasets (not committed to git)
├── notebooks/                  # Numbered, sequential pipeline notebooks (01-15)
├── src/                        # Reusable pipeline code
│   ├── data/                   # Dataset loading
│   ├── preprocessing/          # Cleaning, splitting, scaling, feature selection
│   ├── models/                 # Baseline + advanced model training
│   ├── evaluation/             # Metrics + plotting
│   ├── explainability/         # SHAP utilities
│   ├── cross_dataset/          # Feature alignment + domain shift analysis
│   └── utils.py                # Config loading, reproducibility, plotting setup
├── results/
│   ├── processed/               # Saved train/test splits (parquet)
│   ├── models/                  # Saved trained models (joblib)
│   ├── metrics/                 # Result tables (CSV)
│   ├── figures/                 # Generated plots (PNG)
│   └── reports/                 # JSON reports (cleaning, alignment, domain shift)
├── requirements.txt
└── README.md
```

## Reproducing This Project

### 1. Environment setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Datasets

Download and place datasets according to `configs/config.yaml` paths:

| Dataset | Source | Path |
|---|---|---|
| CICIDS2017 | unb.ca/cic/datasets/ids-2017.html | `datasets/CICIDS2017/MachineLearningCVE/` |
| CSE-CIC-IDS2018 | AWS S3 `s3://cse-cic-ids2018/` | `datasets/CSE_CIC_IDS2018/` |
| UNSW-NB15 | research.unsw.edu.au (pre-split CSVs) | `datasets/UNSW_NB15/` |
| CICDDoS2019 | unb.ca/cic/datasets/ddos-2019.html | `datasets/CICDDoS2019/` |

### 3. Run notebooks in order

Notebooks 01–09 build the core CICIDS2017 pipeline (loading → cleaning → splitting →
baseline models → advanced models → feature selection → final model → evaluation →
explainability). Notebooks 10–11 add the CSE-CIC-IDS2018 cross-dataset test.
Notebooks 12–15 extend cross-dataset testing to UNSW-NB15 and CICDDoS2019.

### 4. Launch the demo UI

```powershell
streamlit run app/app.py
```

## Key Results

| Model | In-Dataset F1 | In-Dataset ROC-AUC |
|---|---|---|
| Logistic Regression | 0.9989 | 0.9999 |
| Decision Tree | 0.9996 | 0.9998 |
| Random Forest (final, tuned, 20 features) | 0.9999 | 1.0000 |
| XGBoost | 0.99998 | 1.0000 |

| Cross-Dataset Target | Feature Overlap | Same Tool | F1-score | FNR |
|---|---|---|---|---|
| CSE-CIC-IDS2018 | 20/20 (100%) | Yes | 0.0000 | 1.0000 |
| UNSW-NB15 | 7/20 (35%) | No | 0.0000 | 1.0000 |
| CICDDoS2019 | 20/20 (100%) | Yes | 0.0000 | 1.0000 |

Full metrics, figures, and reports are saved under `results/`.

### Multi-model cross-dataset comparison
Results for all 4 models on every target are in `results/metrics/multi_model_cross_dataset.csv` and
`threshold_analysis_all_models.csv` (notebook 16). Thresholds in the analysis are chosen using target
labels, so they are diagnostic upper bounds, not deployable results. Note that the CICDDoS2019 file is
98.6% attack, so F1 there is dominated by class prevalence.

## Reproducibility Notes

- Fixed random seed (42) throughout, set via `configs/config.yaml`.
- Train/test split is fit-then-transform: scalers, feature selectors, and models are
  fit exclusively on training data; test/cross-dataset data is only ever transformed.
- Every cleaning, alignment, and domain-shift step logs a machine-readable JSON report
  under `results/reports/` documenting exactly what was removed/mapped/dropped and why.

## Citations for Datasets Used

- Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. (2018). Toward Generating a New
  Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP*.
  (CICIDS2017 / CSE-CIC-IDS2018)
- Moustafa, N., Slay, J. (2015). UNSW-NB15: a comprehensive data set for network
  intrusion detection systems. *MilCIS*, IEEE.
- Sharafaldin, I., Lashkari, A.H., Hakak, S., Ghorbani, A.A. (2019). Developing
  Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy.
  *IEEE 53rd International Carnahan Conference on Security Technology*.
