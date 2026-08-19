"""
Q2 — baseline validation of the feature set.
Purpose: hand the modeling teammate a defensible starting point, i.e.
  (a) which feature blocks actually buy accuracy,
  (b) whether to model SALE_PRICE or log(SALE_PRICE) given the grading metric,
  (c) a CV RMSE benchmark any candidate model has to beat.

Grading metric is RMSE in DOLLARS on 626 held-out homes, so every number below
is dollar RMSE from 10-fold CV on the 2,383 training homes.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold

from build_features import build_all, BASE_NUM, BASE_FLAG

RNG = 42
tr, te, Xtr, ytr, Xte = build_all(slope_dummies=True)
y = ytr.values

nb = [c for c in Xtr.columns if c.startswith("NB_")]
st = [c for c in Xtr.columns if c.startswith("ST_")]
slp = [c for c in Xtr.columns if c.startswith("SLP_")]


def cv_rmse(cols, log_target=False, model="ols", smearing=True, folds=10):
    """10-fold CV, RMSE always reported in dollars."""
    X = Xtr[cols].values
    kf = KFold(n_splits=folds, shuffle=True, random_state=RNG)
    errs = []
    for itr, iva in kf.split(X):
        Xa, Xb = X[itr], X[iva]
        ya, yb = y[itr], y[iva]
        if model == "ols":
            m = LinearRegression()
        else:
            m = RidgeCV(alphas=np.logspace(-3, 3, 25))
        if log_target:
            m.fit(Xa, np.log(ya))
            resid = np.log(ya) - m.predict(Xa)
            # Duan smearing: E[price] = exp(log-pred) * mean(exp(residual)).
            # Without it, exp() of a log-model prediction estimates the MEDIAN,
            # which is biased low for a dollar-RMSE metric.
            factor = np.mean(np.exp(resid)) if smearing else 1.0
            pred = np.exp(m.predict(Xb)) * factor
        else:
            m.fit(Xa, ya)
            pred = m.predict(Xb)
        errs.append(np.mean((pred - yb) ** 2))
    return float(np.sqrt(np.mean(errs)))


specs = [
    ("A. LIVING_SQFT only (level)",          ["LIVING_SQFT"],                         False),
    ("B. all raw features, no dummies",      BASE_NUM + BASE_FLAG,                    False),
    ("C. B + NBHD + STYLE dummies",          BASE_NUM + BASE_FLAG + nb + st,          False),
    ("D. C + slope dummies",                 list(Xtr.columns),                       False),
    ("E. log(price), raw features only",     BASE_NUM + BASE_FLAG,                    True),
    ("F. log(price) + dummies",              BASE_NUM + BASE_FLAG + nb + st,          True),
    ("G. log(price) + dummies + slopes",     list(Xtr.columns),                       True),
]

print("=" * 78)
print("10-fold CV RMSE (dollars) — lower is better")
print("=" * 78)
rows = []
for name, cols, logt in specs:
    r = cv_rmse(cols, log_target=logt)
    rows.append((name, len(cols), r))
    print(f"{name:38s} k={len(cols):3d}   RMSE = ${r:,.0f}")

print("\n--- log-target WITHOUT smearing correction (shows why it matters) ---")
print(f"G, no smearing:                             RMSE = ${cv_rmse(list(Xtr.columns), True, smearing=False):,.0f}")

print("\n--- ridge instead of OLS on the full matrix ---")
print(f"D (level)  + ridge:                         RMSE = ${cv_rmse(list(Xtr.columns), False, model='ridge'):,.0f}")
print(f"G (log)    + ridge:                         RMSE = ${cv_rmse(list(Xtr.columns), True,  model='ridge'):,.0f}")

# ---------------------------------------------------------------------------
# Feature-block ablation on the best family: drop one block, see what breaks.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("ABLATION on spec G (log target, everything) — RMSE change when a block is removed")
print("=" * 78)
full = list(Xtr.columns)
base = cv_rmse(full, True)
blocks = {
    "NBHD dummies":       nb,
    "slope dummies":      slp,
    "STYLE dummies":      st,
    "basement block":     ["FBSMT_SQFT", "LOG_FBSMT", "BSMT_AREA", "LOG_BSMT", "BSMT_FIN_RATIO", "HAS_BSMT", "HAS_FBSMT"],
    "land block":         ["LAND_SQFT", "LOG_LAND", "LAND_PER_LIVING", "HAS_LAND", "SLP_LAND_SFR"],
    "age block":          ["BLDG_AGE", "LOG_AGE", "AGE_SQ", "EFF_AGE", "LOG_EFF_AGE", "IS_NEW", "IS_HISTORIC", "EVER_REMOD", "SLP_AGE_CONDO"],
    "remodel info only":  ["EFF_AGE", "LOG_EFF_AGE", "EVER_REMOD"],
    "bath/bed block":     ["BED_RMS", "FULL_B", "HLF_B", "TOT_BATH", "BATH_PER_BED", "SQFT_PER_ROOM", "IS_STUDIO"],
    "log size transforms":["LOG_LIVING", "LOG_TOT_FIN"],
}
for name, cols in blocks.items():
    keep = [c for c in full if c not in cols]
    r = cv_rmse(keep, True)
    print(f"  drop {name:22s} -> ${r:,.0f}   ({r - base:+,.0f})")

print(f"\n  full spec G baseline:            ${base:,.0f}")

# ---------------------------------------------------------------------------
# Sanity checks the modeling teammate should know about
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("NOTES FOR THE MODELING STEP")
print("=" * 78)
X = Xtr[full].values
print(f"n = {len(X)}, k = {X.shape[1]}  ->  n/k = {len(X)/X.shape[1]:.0f} obs per parameter")
corr = np.corrcoef(np.c_[np.ones(len(X)), X].T)
print(f"duplicate rows (identical features AND price): {tr.drop(columns=['ID']).duplicated().sum()}")
resid_model = LinearRegression().fit(X, np.log(y))
r2 = resid_model.score(X, np.log(y))
print(f"in-sample R^2 of spec G: {r2:.3f}  (vs CV RMSE above — gap = overfitting check)")
