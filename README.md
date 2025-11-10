
# Crypto Trading Bot — Walk-Forward XGBoost Models

This repository implements a modular cryptocurrency prediction pipeline using **feature engineering**, **walk-forward validation with embargo**, and **separate per-coin XGBoost models** for BTC, ETH, and LTC.
It is designed for **live inference** using pre-trained models and an **online feature engine** that accumulates streaming market data.

---

## ⚙️ Project Structure

```
trading_bot/
│
├── data_extraction_feature_engineering.ipynb   # feature generation + preprocessing
├── model_training.ipynb                        # walk-forward training + threshold tuning
├── features.py                                 # offline feature builder
├── online_features.py                          # streaming feature engine (real-time)
├── predict_live.py                             # live model inference
├── test.py                                     # dry-run smoke test
├── models/
│   ├── btc_xgb.joblib
│   ├── eth_xgb.joblib
│   ├── ltc_xgb.joblib
│   ├── feature_columns.joblib
│   ├── thresholds.joblib
│   └── training_results.joblib
└── README.md
```

---

## 🧠 Model Training Pipeline

Training is performed in `model_training.ipynb` using **walk-forward cross-validation with embargo** to simulate realistic time-series learning.

### Steps:

1. **Feature Preparation**

   * Extract rolling and lagged features from historical OHLCV data.
   * Normalize or scale as needed (see `data_extraction_feature_engineering.ipynb`).

2. **Walk-Forward Validation**

   * Splits the dataset into 5 sequential folds using a custom `purged_splits` generator.
   * Each fold:

     * Uses 80% of its training slice for fitting and 20% for validation.
     * Selects the **best threshold** on the validation subset by maximizing **F1-score**.

3. **Global Threshold Selection**

   * After out-of-fold (OOF) predictions are complete, a single optimal threshold per coin is determined from the full OOF results.

4. **Final Model per Coin**

   * Trains one final **XGBoost** classifier for each coin on all available data (with a small validation tail for early stopping).
   * Saves:

     * `btc_xgb.joblib`
     * `eth_xgb.joblib`
     * `ltc_xgb.joblib`

5. **Artifact Saving**

   * `feature_columns.joblib`: the exact order of features expected at inference time.
   * `thresholds.joblib`: per-coin thresholds derived from OOF F1-optimization.
   * `training_results.joblib`: summary of metrics (accuracy, AUC, AP, classification report).

---

## 🔮 Live Prediction

`predict_live.py` loads all trained models and performs real-time predictions from an `OnlineFeatureEngine` instance.

```python
from predict_live import predict_from_tick

tick = {
    "BTC": {...},  # real-time OHLCV or price tick
    "ETH": {...},
    "LTC": {...}
}

result = predict_from_tick(tick)
print(result)
```

Sample output:

```python
{
  "ready": True,
  "BTC": {"proba": 0.976, "label": 1},
  "ETH": {"proba": 0.995, "label": 1},
  "LTC": {"proba": 0.992, "label": 1}
}
```

---

## 🧩 Key Improvements (Nov 2025 Update)

* Fixed bug: identical probabilities caused by saving the same model thrice.
  → Now each coin’s model is trained and saved separately (`models/{coin}_xgb.joblib`).
* Added **OOF threshold optimization** per coin.
* Added **feature_columns.joblib** for strict feature order consistency.
* Added **thresholds.joblib** for deterministic serving thresholds.
* Added **training_results.joblib** for experiment tracking.
* Ensured reproducibility with `random_state=42` across all stages.

---

## 🧾 Example: Running the Full Pipeline

1. **Train new models**

   ```bash
   conda activate Helia
   python -m jupyter notebook model_training.ipynb
   ```

2. **Verify artifacts**

   ```bash
   ls models/
   ```

3. **Test live prediction**

   ```bash
   python test.py
   ```

---

## 📊 Metrics Snapshot

Example (from recent run):

| Coin | Accuracy | ROC-AUC | Avg Precision |
| ---- | -------- | ------- | ------------- |
| BTC  | 0.84     | 0.91    | 0.89          |
| ETH  | 0.86     | 0.92    | 0.90          |
| LTC  | 0.81     | 0.88    | 0.85          |

*(Values illustrative; see `training_results.joblib` for exact numbers.)*

---

## 🧱 Dependencies

* Python ≥ 3.10
* numpy, pandas, scikit-learn
* xgboost
* joblib
* matplotlib (for notebook visualization)

Install:

```bash
pip install -r requirements.txt
```

---

## 🚀 Next Steps

The next development phase focuses on **turning this prediction engine into a complete web application**.

### Planned features:

* **FastAPI backend** exposing endpoints for:

  * `/predict` → returns live model outputs for BTC, ETH, and LTC
  * `/train` (optional, protected) → retrain models on new data
* **Interactive web dashboard** (React or Streamlit) to:

  * Visualize live predictions and confidence levels
  * Plot historical model accuracy and feature importance
  * Display per-coin trading signals and thresholds in real-time
* **Deployment options**:

  * Dockerized app with reproducible environment
  * Optional cloud deployment on **Railway**, **Render**, or **AWS EC2**

The ultimate goal: a browser-based, real-time **crypto prediction web app** where models continuously learn and users can view and interpret live trading signals interactively.
