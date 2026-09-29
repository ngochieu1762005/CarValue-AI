# CarValue AI — Complete Modeling & Web Demo Project

This folder continues the supplied EDA and completes the modeling/evaluation/demo pipeline for the Bonbanh used-car dataset.

## Project target
Predict **online asking price (`price_vnd`)**, not the final negotiated transaction price.

## Folder structure

```text
CarValue_AI_Complete_Project/
├── data/
│   └── bonbanh_usedcar_dataset.csv
├── notebooks/
│   ├── 01_EDA_and_Data_Audit.ipynb
│   ├── 02_Modeling_Data_and_Group_Split.ipynb
│   ├── 03_Baseline_Models.ipynb
│   ├── 04_Advanced_Models_RF_XGB_CatBoost.ipynb
│   ├── 05_Evaluation_Error_Analysis_and_Model_Selection.ipynb
│   ├── 06_Final_Model_Export_and_Web_Preparation.ipynb
│   └── 07_End_to_End_Smoke_Test.ipynb
├── src/
│   └── carvalue_utils.py
├── artifacts/
├── models/
├── figures/
├── assets/
│   └── styles.css
├── app.py
├── requirements.txt
└── run_all_notebooks.py
```

## Recommended notebook order

1. `01_EDA_and_Data_Audit.ipynb` — original research-grade EDA, path-fixed for this project.
2. `02_Modeling_Data_and_Group_Split.ipynb` — creates the repaired modeling dataset and leakage-safe grouped split.
3. `03_Baseline_Models.ipynb` — Median baseline and Ridge regression on log-price.
4. `04_Advanced_Models_RF_XGB_CatBoost.ipynb` — Random Forest, XGBoost, CatBoost.
5. `05_Evaluation_Error_Analysis_and_Model_Selection.ipynb` — MAE/RMSE/R²/RMSLE, tier/brand error, feature importance, final model selection.
6. `06_Final_Model_Export_and_Web_Preparation.ipynb` — creates web dropdown options and verifies form-style prediction.
7. `07_End_to_End_Smoke_Test.ipynb` — final delivery checks.


## Version 2 — Web/demo fixes

This package includes the corrected web demo requested after the first UI test:

- **Model-aware form:** Brand → model → year → technical fields are filtered sequentially using combinations actually observed in the modeling data. Impossible cross-model combinations are no longer offered.
- **EV handling:** engine displacement is always `NaN` for electric cars. The app displays `N/A — xe điện` instead of forcing values such as 2.0 L.
- **Conservative target quality rule:** the repaired master dataset still retains every row for audit, but a listing is excluded from supervised modeling only if its price is more than 4× or less than 1/4 of the median for the same brand-model-year group with at least 4 observations. In this dataset this flags exactly **1** row.
- **Plausibility guardrail:** the UI compares the raw model output against train-only price quantiles for the selected model/year. It only intervenes when a prediction is clearly outside a deliberately wide reference range, and the raw prediction remains visible for audit.
- **Market context:** prediction panel displays comparable-listing count, train median, IQR and a simple data-support indicator.
- **Redesigned UI:** Montserrat typography, responsive cards, cleaner spacing, custom HTML/CSS in `assets/styles.css`.

Regression sanity check included in notebooks 06 and 07:

```text
VinFast VF3 Plus 2025, valid observed configuration
raw prediction: ~242.9M VND
train median:    ~239.5M VND
reference n:     98
```

After the conservative modeling-data repair, the selected Random Forest test metrics are approximately:

```text
MAE  = 68.26 million VND
RMSE = 202.18 million VND
R²   = 0.9751
```

## Install

```bash
pip install -r requirements.txt
```

## Run every notebook automatically

From the project root:

```bash
python run_all_notebooks.py
```

Executed notebooks are overwritten with their outputs so you can submit/show the results directly.

## Launch the web demo

After notebooks 05 and 06 have been run:

```bash
streamlit run app.py
```

## Modeling choices

- Group-aware train/test split by `vehicle_signature` to reduce repeated-record leakage.
- Baseline location is excluded because missingness is structurally tied to the collection process.
- `condition` is excluded because it is constant in this dataset.
- `price_recovered_flag`, `vehicle_signature`, and other provenance fields are not predictive inputs.
- Log-price targets are used for the trainable regressors because the price distribution is strongly right-skewed.
- Final model is selected by the lowest MAE on the unseen grouped test set.

## Main deliverables produced by the notebooks

- `artifacts/CarValue_AI_model_candidate.csv`
- `artifacts/metrics_summary.csv`
- `artifacts/mae_by_price_tier.csv`
- `artifacts/mae_by_major_brand.csv`
- `artifacts/feature_importance_top20.csv`
- `models/final_model.joblib`
- `models/final_model_metadata.json`
- report-ready figures in `figures/`
- Streamlit demo via `app.py`


## Windows runner fix

If `python run_all_notebooks.py` previously failed with `FileNotFoundError: [WinError 2]`, use the updated runner in this package. It calls `python -m nbconvert` through the current Python interpreter, so it does not depend on the `jupyter.exe` command being on PATH.

Windows:
```bat
python run_all_notebooks.py
```
Or double-click `run_project_windows.bat`.
