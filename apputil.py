import numpy as np
import pandas as pd


class GroupEstimate(object):
    def __init__(self, estimate):
        if estimate not in {"mean", "median"}:
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self._cols = None
        self._mapping_full = None
        self._fallback_col = None
        self._mapping_fallback = None

    def fit(self, X, y, default_category=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        y = pd.Series(y, name="_y")
        if y.isna().any():
            raise ValueError("y contains missing values")

        df = X.copy()
        df["_y"] = y.values
        self._cols = list(X.columns)

        if default_category is not None:
            if default_category not in self._cols:
                raise ValueError(f"default_category '{default_category}' not in X columns.")
            self._fallback_col = default_category

        aggfunc = "mean" if self.estimate == "mean" else "median"

        # full-combination mapping
        grouped_full = (
            df.groupby(self._cols, observed=True, dropna=False)["_y"]
              .agg(aggfunc)
        )
        if len(self._cols) == 1:
            self._mapping_full = {(k if isinstance(k, tuple) else (k,)): v for k, v in grouped_full.items()}
        else:
            self._mapping_full = {k: v for k, v in grouped_full.items()}

        # fallback mapping
        if self._fallback_col is not None:
            grouped_fb = (
                df.groupby(self._fallback_col, observed=True, dropna=False)["_y"]
                  .agg(aggfunc)
            )
            self._mapping_fallback = dict(grouped_fb.items())
        else:
            self._mapping_fallback = None

        return self

    def predict(self, X_):
        if self._mapping_full is None:
            raise RuntimeError("Call fit(X, y) before predict().")

        # Normalize input
        if isinstance(X_, pd.DataFrame):
            Xp = X_[self._cols].copy()
        else:
            X_ = np.asarray(X_)
            if X_.ndim == 1:
                if len(self._cols) != 1:
                    raise ValueError(f"Expected {len(self._cols)} columns, got 1D input.")
                X_ = X_.reshape(-1, 1)
            if X_.shape[1] != len(self._cols):
                raise ValueError(f"Expected {len(self._cols)} columns, got {X_.shape[1]}.")
            Xp = pd.DataFrame(X_, columns=self._cols)

        if len(self._cols) == 1:
            keys_full = [(v,) for v in Xp[self._cols[0]].tolist()]
        else:
            keys_full = list(map(tuple, Xp[self._cols].itertuples(index=False, name=None)))

        preds = []
        unseen_combo = 0
        filled_by_fallback = 0

        for i, k in enumerate(keys_full):
            v = self._mapping_full.get(k, np.nan)

            if pd.isna(v):
                unseen_combo += 1
                if self._mapping_fallback is not None:
                    fb_val = Xp.iloc[i][self._fallback_col]
                    v_fb = self._mapping_fallback.get(fb_val, np.nan)
                    if not pd.isna(v_fb):
                        v = v_fb
                        filled_by_fallback += 1
            preds.append(v)

        if unseen_combo:
            msg = f"{unseen_combo} observation(s) belong to unseen combination(s); "
            if self._mapping_fallback is not None:
                msg += f"{filled_by_fallback} filled via fallback='{self._fallback_col}', "
            msg += f"{sum(pd.isna(preds))} remain NaN."
            print(msg)

        # ✅ Return a NumPy array (to satisfy test)
        return np.array(preds, dtype=float)
