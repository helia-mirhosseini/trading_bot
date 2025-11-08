import joblib
from online_features import OnlineFeatureEngine

clf_btc = joblib.load("models/btc_xgb.joblib")
clf_eth = joblib.load("models/eth_xgb.joblib")
clf_ltc = joblib.load("models/ltc_xgb.joblib")
feature_cols = joblib.load("models/feature_columns.joblib")

# optional if you saved them
try:
    th = joblib.load("models/thresholds.joblib")
except:
    th = {"btc": 0.55, "eth": 0.55, "ltc": 0.50}

engine = OnlineFeatureEngine(maxlen=3000)
engine.set_feature_columns(feature_cols)

def predict_from_tick(tick: dict):
    X = engine.update(tick)
    if X is None:
        return {"ready": False}

    p_btc = float(clf_btc.predict_proba(X)[0, 1])
    p_eth = float(clf_eth.predict_proba(X)[0, 1])
    p_ltc = float(clf_ltc.predict_proba(X)[0, 1])

    return {
        "ready": True,
        "BTC": {"proba": p_btc, "label": int(p_btc >= th.get("btc", 0.55))},
        "ETH": {"proba": p_eth, "label": int(p_eth >= th.get("eth", 0.55))},
        "LTC": {"proba": p_ltc, "label": int(p_ltc >= th.get("ltc", 0.50))},
    }
