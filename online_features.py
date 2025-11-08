# online_features.py
import numpy as np
import pandas as pd
from collections import deque
from features import build_features_offline, COINS

class OnlineFeatureEngine:
    """
    Feed with dict rows: {"bitcoin_price":..., "bitcoin_volume":..., ...}
    On each update(), recompute rolling features and return the latest
    past-only feature vector (row - 1) aligned with training.
    """
    def __init__(self, maxlen: int = 3000):
        self.buffer = deque(maxlen=maxlen)
        self.df = pd.DataFrame()
        self._last_feature_cols = None  # set by set_feature_columns()

    def set_feature_columns(self, cols: list[str]):
        self._last_feature_cols = cols

    def update(self, tick: dict) -> pd.DataFrame | None:
        self.buffer.append(tick)
        self.df = pd.concat([self.df, pd.DataFrame([tick])], ignore_index=True)

        # Recompute rolling features over the buffer (fast enough for few thousand rows)
        feat_df = build_features_offline(self.df)

        # We want features at time t to NOT use info from t or future,
        # so we take row - 2 as the "latest fully lagged" feature row,
        # then in serving we’ll still select the training column order.
        if len(feat_df) < 2:
            return None
        latest = feat_df.iloc[-2].copy()

        # Drop raw inputs you didn’t use directly (optional; we’ll reselect columns later anyway)
        drop_raw = []
        for c in COINS:
            drop_raw += [f"{c}_price", f"{c}_volume"]
        latest = latest.drop(labels=[x for x in drop_raw if x in latest.index], errors="ignore")

        X_row = latest.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_frame().T
        if self._last_feature_cols is not None:
            # align column order exactly as training
            missing = [c for c in self._last_feature_cols if c not in X_row.columns]
            for m in missing:
                X_row[m] = 0.0
            X_row = X_row[self._last_feature_cols]
        return X_row
