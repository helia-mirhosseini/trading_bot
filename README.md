
# Multi-Coin Direction Prediction with XGBoost

Predicting short-term **price direction** for multiple cryptocurrencies (Bitcoin, Ethereum, Litecoin) using a fully custom, end-to-end pipeline — from **data extraction** and **feature engineering** to **time-aware XGBoost modeling**.

---

## 🧩 Project Overview

This project builds a supervised learning pipeline to forecast whether each coin’s future cumulative return over the next *H* time steps will be **positive (up)** or **negative (down)**.

It consists of two main stages:

1. **`data_extraction_feature_engineering.ipynb`**

   * Fetches raw historical crypto data (price, volume, etc.)
   * Computes derived features such as moving averages, volatility, and correlations
   * Produces a clean, lagged tabular dataset ready for modeling

2. **`model_training.ipynb`**

   * Creates direction labels (`y_btc`, `y_eth`, `y_ltc`) with no look-ahead bias
   * Trains XGBoost models using purged walk-forward cross-validation with embargo
   * Performs threshold tuning per fold for best precision/recall balance
   * Outputs robust out-of-fold (OOF) metrics for honest evaluation

---

## 📦 Data Extraction & Feature Engineering

The **dataset is entirely self-built**.

### 1. Data Sources

Raw time-series data for:

* Bitcoin (BTC)
* Ethereum (ETH)
* Litecoin (LTC)

Typically includes:

* Price (`*_price`)
* Trading volume (`*_volume`)
* Returns (`*_return`)

### 2. Engineered Features

The feature-engineering notebook constructs dozens of useful predictors, including:

| Feature Type                | Description                           | Example Columns                                |
| --------------------------- | ------------------------------------- | ---------------------------------------------- |
| **Log-scaled volume**       | Stabilizes variance                   | `log_bitcoin_volume`                           |
| **Moving averages (MA)**    | Captures short- and long-term trends  | `bitcoin_ma7`, `bitcoin_ma30`                  |
| **Volatility**              | Rolling standard deviation of returns | `bitcoin_volatility`                           |
| **Lagged returns**          | Last known return at each timestep    | `bitcoin_return_lag1`                          |
| **Cross-coin correlations** | Measures co-movement between assets   | `btc_eth_corr`, `btc_ltc_corr`, `ltc_eth_corr` |

All features are computed using **rolling windows** and **shifted by one time step** to ensure that no future information leaks into model training.

---

## 🧠 Label Construction (No Leakage)

Each coin’s binary target label is defined as:

```
y_coin(t) = 1  if  sum of future returns over next H steps > 0
          = 0  otherwise
```

This captures whether the coin’s short-term direction is upward or downward.

Example in code:

```python
H = 20  # lookahead horizon
fut = df["bitcoin_return"].shift(-H).rolling(H).sum()
df["y_btc"] = (fut > 0).astype(int)
```

---

## ⚙️ Model Training

### Key Steps

1. **Feature Lagging**
   Every feature is shifted by +1 step so that features at time *t* depend only on information available before time *t*.

2. **Purged Walk-Forward Cross-Validation**

   * Time-ordered folds (no random shuffling)
   * An **embargo gap** between train/test folds prevents window overlap leakage

3. **Early Stopping and Threshold Tuning**

   * Each model stops when validation AUC stops improving
   * Best probability threshold is chosen per fold via F1 optimization

4. **Out-of-Fold Evaluation**
   Final metrics are computed across all folds using predictions on unseen data only.

---

## 📈 Example Results (H = 20)

| Coin    | Accuracy | ROC-AUC | Avg Precision |
| ------- | -------- | ------- | ------------- |
| **BTC** | 0.611    | 0.622   | 0.511         |
| **ETH** | 0.621    | 0.675   | 0.557         |
| **LTC** | 0.580    | 0.557   | 0.571         |

These scores indicate **real predictive signal** beyond random chance, especially for Ethereum.
Increasing `H` reduced short-term noise and improved model stability.

---

## 🧮 Metrics Explained

| Metric                | Meaning                                |
| --------------------- | -------------------------------------- |
| **Accuracy**          | Fraction of correct up/down calls      |
| **ROC-AUC**           | Ranking quality, threshold-independent |
| **Average Precision** | Area under precision-recall curve      |
| **F1-score**          | Trade-off between precision and recall |

---

## ⚡ Parameters to Tune

| Parameter          | Description          | Typical Range |
| ------------------ | -------------------- | ------------- |
| `H`                | Lookahead horizon    | 10 – 50       |
| `embargo`          | Gap before test fold | 20 – 100      |
| `scale_pos_weight` | Balance class skew   | 1.0 – 2.0     |
| `max_depth`        | Tree depth           | 5 – 7         |
| `reg_lambda`       | L2 regularization    | 1 – 4         |

---

## 🧰 Requirements

```
python >= 3.9
numpy
pandas
scikit-learn
xgboost
```

Install all dependencies:

```bash
pip install numpy pandas scikit-learn xgboost
```

---

## 🚀 How to Run

1. **Feature Engineering**

   ```bash
   jupyter notebook data_extraction_feature_engineering.ipynb
   ```

   Generates a cleaned dataset `df` with all engineered features.

2. **Model Training**

   ```bash
   jupyter notebook model_training.ipynb
   ```

   * Adjust `H` to control forecast horizon
   * Run all cells to train models for BTC, ETH, and LTC
   * View printed metrics in the output cells

---

## 💡 Tips for Improvement

* Train multiple horizons (H = 10, 20, 50) and **average probabilities** for an ensemble signal.
* Add **trend ratios** (`ma7 / ma30`), **volatility ratios** (`vol7 / vol30`), and **time-of-day encodings**.
* Apply **confidence filtering** — only act when `p > 0.6` or `p < 0.4`.
* Track metrics on **rolling windows** to detect regime changes.

---

## 🧾 Repository Structure

```
.
├── data_extraction_feature_engineering.ipynb   # Builds dataset & features
├── model_training.ipynb                        # Trains and evaluates models
├── data/                                       # Optional folder for raw data
├── README.md                                   # This file
```

---

## 🧠 License & Disclaimer

This repository and dataset were built from scratch for **educational and research purposes**.
Performance will vary by timeframe and market regime.
No guarantee of profitability — use responsibly.

