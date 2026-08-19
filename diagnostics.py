"""Extra diagnostics: where the dollar error comes from, and how much headroom
a non-linear model has. Informational hand-off for the modeling teammate."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from build_features import build_all, BASE_NUM, BASE_FLAG

RNG = 42
tr, te, Xtr, ytr, Xte = build_all(True)
y = ytr.values
full = list(Xtr.columns)
nb = [c for c in full if c.startswith("NB_")]
st = [c for c in full if c.startswith("ST_")]
slp = [c for c in full if c.startswith("SLP_")]


def oof_pred(cols, kind, log_target):
    X = Xtr[cols].values
    kf = KFold(10, shuffle=True, random_state=RNG)
    p = np.zeros(len(X))
    for a, b in kf.split(X):
        t = np.log(y[a]) if log_target else y[a]
        if kind == "ols":
            m = LinearRegression()
        elif kind == "ridge":
            m = RidgeCV(alphas=np.logspace(-3, 3, 25))
        elif kind == "rf":
            m = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                      random_state=RNG, n_jobs=-1)
        else:
            m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                              random_state=RNG)
        m.fit(X[a], t)
        pb = m.predict(X[b])
        if log_target:
            pb = np.exp(pb) * np.mean(np.exp(t - m.predict(X[a])))
        p[b] = pb
    return p


def rmse(p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


print("=" * 74)
print("MODEL FAMILY COMPARISON (10-fold out-of-fold, dollar RMSE)")
print("=" * 74)
cands = {
    "OLS  level  full":  ("ols", full, False),
    "OLS  log    full":  ("ols", full, True),
    "Ridge log   full":  ("ridge", full, True),
    "Ridge log   lean":  ("ridge", BASE_NUM + BASE_FLAG + st + slp, True),
    "RandomForest log":  ("rf", full, True),
    "HistGBM     log":   ("gbm", full, True),
    "HistGBM     level": ("gbm", full, False),
}
preds = {}
for name, (k, cols, lt) in cands.items():
    preds[name] = oof_pred(cols, k, lt)
    print(f"  {name:20s} ${rmse(preds[name]):>10,.0f}")

best_lin = preds["Ridge log   full"]
best_gbm = preds["HistGBM     log"]
print(f"\n  50/50 blend (ridge-log + GBM-log): ${rmse(0.5*best_lin + 0.5*best_gbm):,.0f}")

print("\n" + "=" * 74)
print("WHERE THE DOLLAR ERROR LIVES (ridge-log OOF predictions)")
print("=" * 74)
d = pd.DataFrame({"y": y, "p": best_lin})
d["err"] = d.p - d.y
d["dec"] = pd.qcut(d.y, 10, labels=False) + 1
g = d.groupby("dec").agg(n=("y", "size"), price_lo=("y", "min"), price_hi=("y", "max"),
                         bias=("err", "mean"), rmse=("err", lambda s: np.sqrt((s**2).mean())),
                         mape=("err", lambda s: 0.0))
g["mape_pct"] = d.groupby("dec").apply(lambda x: (x.err.abs()/x.y).mean()*100, include_groups=False)
print(g.drop(columns="mape").round(0).to_string())
share = d.groupby("dec").err.apply(lambda s: (s**2).sum()) / (d.err**2).sum()
print("\nshare of total squared error by price decile:")
print((share*100).round(1).to_string())

print("\nmedian |%| error overall: {:.1f}%  (Zillow's off-market MAPE ~6.9%)".format(
    (np.abs(d.err)/d.y).median()*100))

print("\n" + "=" * 74)
print("OUTLIER SENSITIVITY (train-side only; test rows can never be dropped)")
print("=" * 74)
ppsf = y / tr.LIVING_SQFT.values
lo, hi = np.percentile(ppsf, [0.5, 99.5])
keep = (ppsf > lo) & (ppsf < hi)
X = Xtr[full].values
kf = KFold(10, shuffle=True, random_state=RNG)
errs = []
for a, b in kf.split(X):
    a = a[keep[a]]                      # trim only the TRAINING half of each fold
    m = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X[a], np.log(y[a]))
    f = np.mean(np.exp(np.log(y[a]) - m.predict(X[a])))
    errs.append(np.mean((np.exp(m.predict(X[b]))*f - y[b])**2))
print(f"  ridge-log, 0.5/99.5% $/sqft trim on train folds: ${np.sqrt(np.mean(errs)):,.0f}")
print(f"  ridge-log, no trim:                              ${rmse(best_lin):,.0f}")
print(f"  rows trimmed: {(~keep).sum()}")
