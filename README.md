# Q2 — Data cleaning & feature engineering

Owner: **Q2 data/features**. This folder produces the feature matrix that the
Q2 modeling step consumes. Nothing here picks a final model.

## Quick start

```bash
pip install -r requirements.txt
python build_features.py          # writes train_features.csv / test_features.csv
```

Or just import it — this is what the modeling step should do:

```python
from build_features import build_all

tr, te, Xtr, ytr, Xte = build_all(slope_dummies=True)
#  Xtr : 2383 x 69   ytr : SALE_PRICE in dollars
#  Xte :  626 x 69   identical columns, identical order
#  te['ID'] is the key for writing PREDICTED_PRICE back into DenverRE_Test.xlsx
```

## Files

| File | What it is |
|---|---|
| `build_features.py` | The whole pipeline: `clean()` → `engineer()` → `design_matrix()`. Train and test go through the **same** function, so there is no train/test skew. |
| `feature_dictionary.csv` | Every column: definition + why it exists. Read this first. |
| `train_features.csv` / `test_features.csv` | Pre-built output, if you'd rather not run the script. |
| `validate_features.py` | 10-fold CV comparison of feature sets + leave-one-block-out ablation. |
| `diagnostics.py` | Model-family comparison, error decomposition by price decile, outlier sensitivity. |
| `topdecile.py` | Focused tests on the expensive homes, where most of the dollar error lives. |
| `data/` | The three course xlsx files, so the code runs out of the box. |

## What the cleaning does (and why)

**`RM_AGE` blanks are not missing data.** Blank for 60.1% of training rows. The data
dictionary says blank = never remodeled, and the data agrees: median building age is
4 years for the blank group vs 49 years for the rest. So instead of dropping or
imputing, we build `EVER_REMOD` (indicator) and `EFF_AGE` = years since the home was
last new or renovated.

**Several zeros are structural.** `LAND_SQFT` is exactly 0 for 100% of the 730 condos
(a condo owns no land). `BSMT_AREA` is 0 for 54.7%, `GRD_AREA` for 95.0%. We keep the
zeros, add `HAS_*` presence flags (so "none at all" gets its own level shift, separate
from the slope on "how much"), and use `log(1+x)` so the zeros survive the transform.

**Price and size are right-skewed.** `SALE_PRICE` skew 1.43 → −0.07 in logs.
`LIVING_SQFT` 1.17 → −0.03. Log-log gives elasticities instead of a constant
dollars-per-square-foot assumption.

**Location dominates, and not just as a level shift.** Median price per living sqft
runs from $175 (Windsor) to $615 (Union Station) — 3.5×. So we add *slope dummies*
(neighborhood × `log LIVING_SQFT`) on top of ordinary neighborhood dummies.

Other decisions: 12 style codes collapsed to 6 groups (smallest group goes from 3 obs
to 106); k−1 dummies with explicit reference levels (Gateway/Green Valley Ranch,
1-story) to avoid the dummy-variable trap; zero-bedroom studios (32) and zero-age new
builds (259) kept with flags; no feature touches `SALE_PRICE`, so no leakage.

## Benchmarks (10-fold CV, **dollar** RMSE — same metric as the grade)

| Specification | k | CV RMSE |
|---|---|---|
| `LIVING_SQFT` only (= the Q1 model) | 1 | $201,080 |
| All raw attributes, no dummies | 35 | $166,141 |
| + neighborhood & style dummies | 53 | $123,796 |
| + slope dummies | 69 | $111,783 |
| log(price), full matrix, OLS | 69 | $111,762 |
| **log(price), full matrix, ridge** | 69 | **$110,813** |
| ridge + piecewise size terms | 73 | $110,282 |
| HistGradientBoosting, log target | 69 | $105,490 |
| 40% ridge + 60% GBM | — | $100,496 |

Feature engineering cuts CV error 45% versus the single-variable model of Q1.

**Ablation** (drop one block from the full spec, see what breaks):

| Block removed | Δ RMSE |
|---|---|
| age block | **+$5,756** |
| slope dummies | **+$3,849** |
| bath/bed block | +$978 |
| basement block | +$819 |
| remodel info | +$477 |
| log size transforms | +$202 |
| land block | +$67 |
| neighborhood *intercept* dummies | **−$1,024** (better without) |
| style dummies | −$48 (no effect) |

## Five things the modeling step should not skip

1. **Select on dollar RMSE**, not R² or log-scale error. A model that wins in logs can
   lose in dollars.
2. **If you fit log(price), apply the Duan smearing factor** `mean(exp(residual))` from
   the *training* fold before converting back. `exp(prediction)` alone estimates the
   median and is biased low. Worth ~$370 here — small, but free.
3. **The `NB_*` intercept dummies are redundant** once `SLP_*` is in — dropping them
   *improves* CV RMSE by $1,024. Either drop them or use ridge (ridge beats OLS by ~$950
   for exactly this reason).
4. **48% of squared error comes from the top price decile** ($801,900+), where the model
   under-predicts by ~$146k on average. Cheap win: `sample_weight = y` drops top-decile
   RMSE from $246k to $228k. The root cause is that the data has no finish-quality,
   condition, or view variable — the very things that make an expensive home expensive.
5. **Don't drop training outliers.** Tested: trimming the extreme 0.5%/99.5% of $/sqft
   moves CV RMSE by $45. And test rows can't be dropped anyway — all 626 need a price.

## Known caveats

- 19 training rows are exact duplicates once `ID` is ignored. They slightly inflate CV
  accuracy. Not worth removing, but worth a sentence in the write-up.
- Median absolute percentage error is 9.2%, between Zillow's reported 1.9% (on-market)
  and 6.9% (off-market). Reasonable for 15 raw attributes and no comparable sales.
- Whatever model wins, **every team member has to be able to explain it in class.** GBM
  is ~$10k better than ridge, but ridge sits entirely inside the course material
  (dummies, slope dummies, log transforms, shrinkage). That's a team call.
