# Methodology

## Data Sources

| File | Rows | Description |
|---|---|---|
| `final_demand.csv` | 1,240 | SKU–region rows with 12 monthly sales columns + product attributes |
| `detailed_attributes.csv` | 220 | Style-level records used to fill attribute gaps |

Merged on `StyleCode` after standardizing casing in both files.

---

## Data Cleaning

| Issue | Scale | Fix |
|---|---|---|
| Monthly sales nulls | Up to 512/month | Filled with 0 — SKU not yet in market |
| Size typos (3.0, 5.0) | 25 rows | Corrected to 53.0 and 55.0 |
| Sizes < 40mm | 25 rows | Filtered out as invalid |
| ColorDescription nulls | 256 rows | Filled with "Unknown" — excluded from modeling |
| FrameConstruction / Material2 nulls | 6 rows each | Patched via attributes file merge |
| Casing mismatch | All categoricals | Standardized to Title Case across both files |

---

## Data Transformation

**Wide → Long format:** 12 monthly columns melted into one row per SKU–Region–Month — **14,880 rows** total.

**Critical:** rows sorted by `StyleCode`, `Region`, `Month` *before* any feature engineering. Lag features computed on unsorted data produce silently wrong values.

**Log-transform:** `log(1 + Sales)` applied to the target. Raw sales: mean=61, max=1,227, skewness=2.56. Without this, models overfit to outlier SKUs. Predictions reversed with `expm1()` for evaluation.

---

## Feature Engineering — 33 Features

All features use `.shift(1)` or earlier — no future information leaks into training.

### Lag Features
`Lag_1`, `Lag_2`, `Lag_3`, `Lag_6` — sales from 1, 2, 3, 6 months prior

### Rolling Averages
`Rolling_3mo_Avg`, `Rolling_6mo_Avg` — via `.shift(1).rolling(n).mean()`

### Expanding Statistics
`SKU_Avg_Sales`, `SKU_Max_Sales`, `Cumulative_Sales` — via `.shift(1).expanding()`

### Trend & Volatility
`Sales_StdDev` — 3-month rolling std  
`MoM_Growth` — month-over-month % change, clipped ±200%  
`Lag1_Brand_Avg` — brand-level average of lagged sales  
`SKU_vs_Brand_Avg` — this SKU's lag relative to brand peers

### Time Features
`Month_Number`, `Quarter`, `Year`  
`Is_Peak_Season` — binary flag for Jan, Feb, Mar (insurance renewal)

### Price Features
`LogPrice`, `PriceTier`, `MarginRatio`, `Price_vs_Brand_Avg`, `Is_Core_Size` (52–55mm flag)

### Encoded Attributes
`Brand`, `Region`, `Gender`, `FrameShape`, `Material1`, `SunOptical`, `FrameConstruction`, `RXAble` — label encoded

> **`StyleCode` dropped intentionally.** Retaining it inflated R² from 0.52 → 0.69 by memorizing SKU identities. A model that knows which SKU it's predicting cannot generalize to new or untested styles.

---

## Train / Test Split

| Set | Period | Rows |
|---|---|---|
| Train | Sep 2023 – May 2024 | 10,287 |
| Test | Jun – Aug 2024 | 3,720 |

Random splitting was not used — it allows future months to inform past predictions, producing inflated accuracy. Rows missing `Lag_1` through `Lag_3` (first 1–3 months of each SKU's life) were dropped, reducing 14,880 to 14,007 rows. Remaining `Lag_6` nulls (873 rows) imputed with `SimpleImputer(strategy='mean')`.

---

## Models

### Linear Regression (Baseline)
Performance floor. Negative raw-unit R² (−18.0) caused by extreme predictions on outlier SKUs — confirms necessity of log-transform and non-linear modeling.

### Random Forest
300 trees, max depth 15, min samples leaf 2. Strong ensemble benchmark at R²=0.520.

### XGBoost (Primary)
Tuned via `RandomizedSearchCV`, 40 combinations, 3-fold CV, optimizing log-scale R².

**Best parameters:**
```python
{
    'n_estimators': 800,
    'max_depth': 4,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0.1,
    'reg_lambda': 3,
    'reg_alpha': 0
}
```

Shallow depth (4) + high lambda (3) = appropriate regularization for ~10,000 training rows.

---

## Evaluation

| Model | R² (Log) | R² (Raw) | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.310 | −18.009 | 55.6 | 337.3 |
| Random Forest | 0.520 | 0.514 | 26.9 | 53.9 |
| XGBoost (baseline) | 0.580 | 0.523 | 26.9 | 53.5 |
| **XGBoost (tuned)** | **0.737** | **0.542** | **26.4** | **57.8** |

**Log-scale R² is primary** — the model was trained on `log(Sales+1)` and raw-unit R² is unfairly penalized by high-volume outlier SKUs.  
**MAE of 26.4** is directly actionable: buffer initial orders by ~25–30 units per SKU per month.  
**CV R² of 0.738** vs test R² of 0.737 — stable, no significant overfitting.
