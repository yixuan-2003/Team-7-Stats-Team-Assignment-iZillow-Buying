"""The top price decile carries ~49% of total squared error. Since the grade is
dollar RMSE, that is where feature work pays. This script tests feature-side
fixes for it."""

import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from build_features import build_all

RNG = 42
tr, te, Xtr, ytr, Xte = build_all(True)
y = ytr.values
full = list(Xtr.columns)


def add_spline(X, src):
    """Piecewise-linear (hinge) terms in log living area: lets the size gradient
    bend at the knots instead of being one straight line."""
    X = X.copy()
    v = src["LOG_LIVING"].values
    for q in [0.25, 0.5, 0.75, 0.9]:
        k = np.quantile(np.log(tr.LIVING_SQFT.values), q)
        X[f"HINGE_{int(q*100)}"] = np.maximum(v - k, 0)
    return X


def oof(Xdf, kind="ridge", log_target=True, weights=None):
    X = Xdf.values
    kf = KFold(10, shuffle=True, random_state=RNG)
    p = np.zeros(len(X))
    for a, b in kf.split(X):
        t = np.log(y[a]) if log_target else y[a]
        w = None if weights is None else weights[a]
        m = (RidgeCV(alphas=np.logspace(-3, 3, 25)) if kind == "ridge"
             else HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=RNG))
        m.fit(X[a], t, sample_weight=w) if w is not None else m.fit(X[a], t)
        pb = m.predict(X[b])
        if log_target:
            pb = np.exp(pb) * np.mean(np.exp(t - m.predict(X[a])))
        p[b] = pb
    return p


def report(name, p):
    r = np.sqrt(np.mean((p - y) ** 2))
    dec = pd.qcut(y, 10, labels=False)
    top = dec == 9
    rt = np.sqrt(np.mean((p[top] - y[top]) ** 2))
    print(f"  {name:42s} RMSE ${r:>9,.0f}   top-decile ${rt:>9,.0f}  bias ${np.mean(p[top]-y[top]):>9,.0f}")
    return r


print("=" * 92)
print("FIXING THE TOP DECILE — all 10-fold OOF, dollar RMSE")
print("=" * 92)
Xb = Xtr[full]
report("ridge, log target (current best linear)", oof(Xb))
report("ridge, LEVEL target", oof(Xb, log_target=False))
Xs = add_spline(Xb, tr)
report("ridge, log + size hinges", oof(Xs))
report("ridge, level + size hinges", oof(Xs, log_target=False))
# weight each obs by price so dollar errors on expensive homes count more
report("ridge, log + hinges, price-weighted", oof(Xs, weights=y / y.mean()))
report("GBM, log + hinges", oof(Xs, kind="gbm"))
pg = oof(Xs, kind="gbm"); pr = oof(Xs)
report("blend 50/50 ridge-log-hinges + GBM", 0.5 * pr + 0.5 * pg)
for w in [0.3, 0.4, 0.6]:
    report(f"blend {w:.0%} ridge / {1-w:.0%} GBM", w * pr + (1 - w) * pg)
