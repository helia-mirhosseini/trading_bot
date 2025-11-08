import pandas as pd
import numpy as np 


COINS = ["bitcoin", "ethereum", "litecoin"]


MAR_SHORT = 7
MAR_LONG = 30
VOL_WIN = 7
CORR_WIN = 14


def _safe_log1p(s):
    return np.log1p(s.replace([np.inf, -np.inf], np.nan))


def build_features_offline(df:pd.DataFrame) ->pd.DataFrame:
    """
    Replicates the exact features used in training. This MUST match the notebook.
    Produces (per coin):
      - <coin>_return
      - <coin>_ma7, <coin>_ma30
      - <coin>_volatility
      - log_<coin>_volume
      - <coin>_return_lag1
    Cross-coin:
      - btc_eth_corr, btc_ltc_corr, ltc_eth_corr  (correlations of returns)
    """
    df = df.copy()
    for c in COINS: 
        pcol =  f"{c}_price"
        vcol = f"{c}_volume"
        rcol = f"{c}_return"


        # return
        df[rcol] = df[pcol].pct_change()

        # moving average price(no return)
        df[f"{c}_ma{MAR_SHORT}"] = df[pcol].rolling(MAR_SHORT, min_periods=MAR_SHORT).mean()
        df[f"{c}_ma{MAR_LONG}"] = df[pcol].rolling(MAR_LONG, min_periods=MAR_LONG).mean()

        # volatility (no return)
        df[f"{c}_volatility"] = df[rcol].rolling(VOL_WIN, min_periods=VOL_WIN).std()


        # log volume
        df[f"log_{vcol}"] = _safe_log1p(df[vcol])

        # lagged return
        df[f"{c}_return_lag1"] = df[rcol].shift(1)

# cross-coin rolling correlation on returns
    if {"bitcoin_return", "ethereum_return"}.issubset(df.columns):
        df["btc_eth_corr"]  = df['bitcoin_return'].rolling(CORR_WIN, min_periods=CORR_WIN).corr(df['ethereum_return'])
    if {"bitcoin_return", "litecoin_return"}.issubset(df.columns):
        df["btc_ltc_corr"]  = df['bitcoin_return'].rolling(CORR_WIN, min_periods=CORR_WIN).corr(df['litecoin_return'])
    if {"ethereum_return", "litecoin_return"}.issubset(df.columns):
        df["eth_ltc_corr"]  = df['ethereum_return'].rolling(CORR_WIN, min_periods=CORR_WIN).corr(df['litecoin_return'])

    # clean bad values
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def label_directions(df:pd.DataFrame, H:int) -> pd.DataFrame:
    """
    Build binary labels: y_btc, y_eth, y_ltc from future cumulative returns over H steps.
    y_coin = 1 if sum(ret[t+1 ... t+H]) > 0 else 0
    """
    df = df.copy()
    for c in COINS: 
        fut = df[f"{c}_return"].shift(-H).rolling(-H, min_periods= H).sum()
        df[f"y_{c[:3]}"] = (fut >0).astype(int)
    
    return df

def coerce_numeric(X:pd.DataFrame) -> pd.DataFrame: 
    X = X.copy()
    for col in X.columns: 
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors= 'coerce')
    return X.replace([np.inf, -np.inf], np.nan).fillna(0,0)


def feature_column_from(df: pd.DataFrame) -> list[str]:
    """
    Decide which columns are features. We exclude labels here; you can also explicitly
    whitelist columns if you prefer tighter control.
    """

    label_cols = [f"y_{c[:3]}" for c in COINS]
    return [c for c in df.columns if c not in label_cols]

def finilize_training_frame(raw_df: pd.DataFrame, H:int):
    """
    Full pipeline for training:
      1) compute features (same as production)
      2) build labels
      3) lag ALL features by +1 to prevent lookahead
      4) align and drop NaNs
      5) coerce numeric
    Returns:
      X (DataFrame), y_dict (dict of np.array per coin), feature_cols (list)
    """

    # 1) features
    feat_df = build_features_offline(raw_df)

    #  2)labels
    lab_df = label_directions(feat_df, H)
    
    # 3)lag all non-label columns by 1
    label_col = [f"y_{c[:3]}" for c in COINS]
    X_all = lab_df.drop(columns=label_col, errors='ignore').shift(1)

    #  4)slign and drop NaNs
    data = pd.concat([X_all, lab_df[label_col]], axis=1).dropna()

    # 5)split, coerce
    X = coerce_numeric(data.drop(columns=label_col))
    y_dict = {c[:3]: data[f"y_{c[:3]}"].astype(int).values for c in COINS}
    feature_cols = list(X.columns)
    return X, y_dict, feature_cols