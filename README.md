# VSP Vision — Frame Assortment Intelligence

> Demand forecasting + product management analysis for Visionworks eyewear retail  
> **XGBoost · Python · 162 SKUs · 3 Brands · 2 Regions · Sep 2023–Aug 2024**

---

## Overview

An end-to-end demand forecasting pipeline built for Visionworks frame assortment planning. Starting from raw monthly sales data, this project enriches product attributes, engineers 33 temporal features, trains and compares three ML models, and translates model output into 14 concrete, data-backed product management recommendations.

The goal: replace intuition-driven assortment decisions with a repeatable, leakage-free framework that predicts SKU-level monthly demand and surfaces where the assortment has structural gaps.

---

## Results

| Metric | Value |
|---|---|
| Model | XGBoost (Tuned) |
| R² — Log Scale *(primary)* | **0.695** |
| R² — Raw Units | 0.542 |
| Cross-Validated R² | 0.738 |
| MAE | 26.4 units / month |
| Features | 33 |
| Training period | Sep 2023 – May 2024 |
| Test period | Jun – Aug 2024 |
| Total units analyzed | 910,853 |

---

## Project Structure

```
vsp-frame-assortment/
│
├── data/
│   ├── final_demand.csv           # SKU × Region × 12 monthly sales columns
│   └── detailed_attributes.csv   # Style-level product attributes
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb    # Merging, cleaning, wide → long reshape
│   ├── 02_eda.ipynb              # Exploratory analysis, shape/price/brand breakdowns
│   └── 03_xgboost_model.ipynb   # Feature engineering, model training, evaluation
│
├── outputs/
│   ├── vsp_report.html           # Full interactive HTML analysis report
│   └── action_items.docx         # 14 prioritized PM action items with data justification
│
├── docs/
│   └── methodology.md            # Full methodology write-up
│
├── portfolio.html                 # Standalone portfolio case study page
├── requirements.txt
├── .gitignore
└── README.md
```

---

## The Business Problems

**Problem 1 — The $150–$199 tier is over-assorted.**  
64 SKUs (39% of the assortment) concentrated in a single price band generating only $4,017/SKU annually — 37% below the $200–$249 sweet spot at $6,326/SKU.

**Problem 2 — Female patients are structurally underserved.**  
Female-labeled SKUs account for just 15.8% of sales despite having identical per-SKU productivity to male SKUs ($4,783 vs $4,817). The gap is supply, not demand.

---

## Methodology

### Data Preparation
- Two source files merged on `StyleCode`: demand file (1,240 SKU–region rows × 12 monthly columns) + attributes file (220 styles)
- Cleaned 3,055 null values: size typos (3.0 → 53mm), casing mismatches, missing attributes
- Reshaped to long format: **14,880 rows** (one per SKU–Region–Month)
- Log-transformed the sales target `log(1 + Sales)` — raw skewness = 2.56
- Time-based split: train Sep 2023–May 2024 / test Jun–Aug 2024

### Feature Engineering — 33 Features, All Leakage-Free

| Category | Features |
|---|---|
| Lag | Sales 1, 2, 3, 6 months prior |
| Rolling averages | 3-month and 6-month rolling mean |
| Expanding statistics | SKU mean, max, cumulative (since launch) |
| Trend | MoM growth (clipped ±200%), volatility, SKU vs brand peers |
| Time | Month, quarter, year, Jan–Mar peak-season flag |
| Price | Log price, price tier, margin ratio, price vs brand average |
| Attributes | Brand, Region, Gender, Shape, Material, SunOptical (label encoded) |

> **Key decision:** `StyleCode` was dropped. Keeping it inflated R² from 0.52 → 0.69 by memorizing SKU identities rather than learning demand patterns. A model that knows which SKU it's predicting cannot generalize to new styles.

### Model Comparison

| Model | R² (Log) | R² (Raw) | MAE |
|---|---|---|---|
| Linear Regression | 0.310 | −18.0 | 55.6 |
| Random Forest | 0.520 | 0.514 | 26.9 |
| XGBoost (baseline) | 0.580 | 0.523 | 26.9 |
| **XGBoost (tuned)** | **0.695** | **0.542** | **26.4** |

Tuned via `RandomizedSearchCV` across 40 combinations, 3-fold CV.  
Best params: `n_estimators=800, max_depth=4, learning_rate=0.01, subsample=0.7, reg_lambda=3`

---

## Key Findings

**What drives demand:** Sales trajectory beats all static product attributes. Top 5 features by importance: Rolling 6-Month Avg Sales → SKU Max Sales → SKU Avg Sales → Brand → Optical/Sun. Shape, material, and size have weak individual signal but strong interaction effects.

**Assortment gaps:**
- **Oval frames** — 1 SKU, 9,432 units, highest avg/SKU of any shape. Proven demand, no supply.
- **Titanium** — 5 SKUs at $9,852/SKU (2× Acetate). NIKE 8045 and 8154 are both top-10 SKUs.
- **Cat Eye** — 2 SKUs total. Nationally growing trend. Visionworks has no real presence.
- **CK Sun** — 80% dead or slow SKUs, $1,197/SKU. Worst performing brand in portfolio.
- **Lacoste in AMER** — 37.6% of EMEA but only 14.6% of AMER. Their $200–$249 tier = $10,005/SKU.

**Seasonality:** Sales jump 73–80% from December to March (insurance plan resets). New styles must be floor-ready by December 15 or miss ~40% of their first-season window.

---

## Top 10 SKUs

| Rank | Style | Brand | Region | Shape | Material | Price | Units |
|---|---|---|---|---|---|---|---|
| 1 | L2707 | Lacoste Optical | EMEA | Square | Acetate | $199.95 | 20,417 |
| 2 | L2876 | Lacoste Optical | EMEA | Square | Acetate | $220.00 | 14,317 |
| 3 | NIKE 7125 | Nike Optical | AMER | Rectangle | Bio Inj-G820 | $249.00 | 14,219 |
| 4 | NIKE 7246 | Nike Optical | AMER | Rectangle | Acetate | $259.00 | 14,096 |
| 5 | L2707N | Lacoste Optical | EMEA | Square | Acetate | $199.95 | 13,443 |
| 6 | NIKE 8045 | Nike Optical | AMER | Rectangle | Titanium | $295.00 | 13,048 |
| 7 | L2271 | Lacoste Optical | EMEA | Rectangle | Metal | $225.00 | 12,086 |
| 8 | NIKE 8154 | Nike Optical | AMER | Rectangle | Titanium | $295.00 | 11,767 |
| 9 | NIKE 7090 | Nike Optical | AMER | Square | Bio Inj-G820 | $285.00 | 11,583 |
| 10 | NIKE 5538 | Nike Optical | AMER | Square | Bio Inj-G820 | $135.00 | 11,129 |

---

## Priority Action Items

| # | Priority | Action | Justification |
|---|---|---|---|
| 1 | 🔴 HIGH | Cut 10–15 SKUs from $150–$199 tier | $4,017/SKU vs $6,326/SKU — 37% gap |
| 2 | 🔴 HIGH | Add 4–6 Lacoste styles to AMER ($200–$249) | $10,005/SKU — best brand×tier combo |
| 3 | 🔴 HIGH | Floor Hero SKUs by Dec 15 | 73–80% sales jump Jan–Mar |
| 4 | 🔴 HIGH | Expand Titanium to 8–10 SKUs | $9,852/SKU, nearly 2× Acetate |
| 5 | 🟡 MED | Cut CK Sun from 15 → 5–6 SKUs | $1,197/SKU, 80% dead |
| 6 | 🟡 MED | Reduce Lacoste Suns; EMEA-only | 17/18 SKUs dead, 84% EMEA |
| 7 | 🟡 MED | Add female-specific Nike + CK styles | Equal productivity, 30 vs 69 SKUs |
| 8 | 🟡 MED | Build dedicated EMEA buy list | Current top-10 is 9/10 Nike AMER |
| 9 | 🟡 MED | Separate AMER/EMEA size grids | AMER: 54mm peak, EMEA: 53mm peak |
| 10 | 🟡 MED | Rationalize CK Optical to 20–25 SKUs | $3,809/SKU — lowest optical brand |
| 11 | 🔵 LOW | Test 2–3 Oval optical styles | 9,432 avg/SKU — highest of any shape |
| 12 | 🔵 LOW | Test 2–3 Cat Eye styles | 2 SKUs only, growing national trend |
| 13 | 🔵 LOW | Extend model to 24+ months of history | +0.04–0.08 R² estimated gain |
| 14 | 🔵 LOW | Implement Dec 15 launch calendar | Late arrivals miss 40% of seasonal peak |

---

## Setup

```bash
git clone https://github.com/yourusername/vsp-frame-assortment.git
cd vsp-frame-assortment
pip install -r requirements.txt
```

Open notebooks in order:

```
notebooks/01_data_cleaning.ipynb
notebooks/02_eda.ipynb
notebooks/03_xgboost_model.ipynb
```

---

## Tech Stack

`Python 3.11` `pandas` `numpy` `XGBoost 2.0` `scikit-learn` `matplotlib` `Jupyter`

---

*Graduate-level data analytics project — VSP Vision / Visionworks frame assortment analysis.*
