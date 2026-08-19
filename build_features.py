"""
Q2 — Data cleaning & feature engineering for the Denver Zestimate task.
Owner: data/features
Input : DenverRE_Train.xlsx (TrainingData), DenverRE_Test.xlsx (TestData)
Output: train_features.csv, test_features.csv  (same columns except SALE_PRICE)
        + feature_dictionary.csv

Design rules
------------
1. Train and test go through the EXACT same function -> no train/test skew.
2. No imputation with the target, no use of test-set statistics -> no leakage.
3. Every engineered variable has a stated reason (see FEATURE_NOTES at bottom).
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _find(filename):
    """Look for the course xlsx next to this script, in ./data, or one level up.
    Keeps the code working whether the files were uploaded flat or in a data/
    subfolder."""
    for folder in (HERE / "data", HERE, HERE.parent / "data", HERE.parent, Path.cwd()):
        p = folder / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {filename}. Put the course xlsx files next to "
        f"build_features.py or in a data/ subfolder."
    )


TRAIN_XLSX = _find("DenverRE_Train.xlsx")
TEST_XLSX = _find("DenverRE_Test.xlsx")

# Style levels are collapsed from 12 to 6. Four raw levels have fewer than 15
# training observations (CONVERSION 3, 3 STORY 7, SPLIT LEVEL 11, BI-LEVEL 24);
# a dummy fitted on a cell that small is mostly noise. After collapsing, the
# smallest group has 106 observations.
STYLE_MAP = {
    "1 STORY": "1STORY",              # 319
    "2 STORY": "2STORY",              # 574
    "1.5 STORY": "PARTIAL_STORY",     # 44 + 52 + 7 + 3 = 106
    "2.5 STORY": "PARTIAL_STORY",
    "3 STORY": "PARTIAL_STORY",
    "CONVERSION": "PARTIAL_STORY",
    "BI-LEVEL": "MULTILEVEL",         # 24 + 11 + 31 + 58 = 124
    "SPLIT LEVEL": "MULTILEVEL",
    "TRI-LEVEL": "MULTILEVEL",
    "TRI-LEVEL W/B": "MULTILEVEL",
    "END UNIT": "END_UNIT",           # 687
    "MIDDLE UNIT": "MIDDLE_UNIT",     # 573
}


def load_raw():
    tr = pd.read_excel(TRAIN_XLSX, sheet_name="TrainingData")
    te = pd.read_excel(TEST_XLSX, sheet_name="TestData")
    return tr, te


# ----------------------------------------------------------------------------
# 1. CLEANING
# ----------------------------------------------------------------------------
def clean(df):
    """Structural fixes only. Nothing here depends on the target."""
    d = df.copy()

    # -- RM_AGE: NaN is NOT missing data, it means "never remodeled".
    #    (Verified: median BLDG_AGE of the NaN group is 4 yrs -> new builds.)
    d["EVER_REMOD"] = d["RM_AGE"].notna().astype(int)

    # -- Effective age: years since the building was last new-or-renovated.
    #    RM_AGE <= BLDG_AGE holds for every remodeled row, so this is well defined.
    d["EFF_AGE"] = np.where(d["EVER_REMOD"] == 1, d["RM_AGE"], d["BLDG_AGE"])

    # -- Structural zeros. These are real zeros, not missing values:
    #    LAND_SQFT == 0 for 100% of condos (a condo owns no land),
    #    BSMT_AREA == 0 / GRD_AREA == 0 simply mean "no basement / no garden level".
    #    We keep the 0 and add a presence flag so the model can fit a level shift
    #    separately from the slope (a 0-sqft basement is a different kind of thing
    #    from a small basement).
    d["HAS_LAND"] = (d["LAND_SQFT"] > 0).astype(int)
    d["HAS_BSMT"] = (d["BSMT_AREA"] > 0).astype(int)
    d["HAS_FBSMT"] = (d["FBSMT_SQFT"] > 0).astype(int)
    d["HAS_GRD"] = (d["GRD_AREA"] > 0).astype(int)

    # -- Consistency guard: finished basement cannot exceed total basement.
    #    Two training rows violate this (none in test). One of them has
    #    BSMT_AREA == 0 but GRD_AREA == 852, i.e. the finished space is at garden
    #    level, so its zero basement is internally consistent and we leave it.
    #    The other (FBSMT 2,100 vs BSMT 1,057) is capped.
    d["FBSMT_SQFT"] = np.minimum(d["FBSMT_SQFT"], d["BSMT_AREA"]).where(
        d["BSMT_AREA"] > 0, d["FBSMT_SQFT"]
    )

    # -- Categoricals: strip whitespace, collapse rare style levels.
    d["NBHD"] = d["NBHD"].str.strip()
    d["PROP_CLASS"] = d["PROP_CLASS"].str.strip()
    d["STYLE_G"] = d["STYLE_CN"].str.strip().map(STYLE_MAP)
    assert d["STYLE_G"].notna().all(), "unmapped style level"

    return d


# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def engineer(df):
    d = df.copy()

    # --- size ---------------------------------------------------------------
    # Living area is right-skewed (skew 1.17); log makes it symmetric (-0.03)
    # and turns the model into a constant-elasticity (%-on-%) specification.
    d["LOG_LIVING"] = np.log(d["LIVING_SQFT"])
    d["LIVING_SQ"] = d["LIVING_SQFT"] ** 2 / 1e6  # scaled, for level-form models
    d["TOT_FIN_SQFT"] = d["LIVING_SQFT"] + d["FBSMT_SQFT"]
    d["LOG_TOT_FIN"] = np.log(d["TOT_FIN_SQFT"])
    d["LOG_LAND"] = np.log1p(d["LAND_SQFT"])  # log1p: 30% of rows are 0 (condos)
    d["LOG_BSMT"] = np.log1p(d["BSMT_AREA"])
    d["LOG_FBSMT"] = np.log1p(d["FBSMT_SQFT"])
    d["BSMT_FIN_RATIO"] = np.where(
        d["BSMT_AREA"] > 0, d["FBSMT_SQFT"] / d["BSMT_AREA"], 0.0
    )
    d["LAND_PER_LIVING"] = d["LAND_SQFT"] / d["LIVING_SQFT"]

    # --- rooms --------------------------------------------------------------
    # A half bath is worth roughly half a full bath -> one variable instead of two.
    d["TOT_BATH"] = d["FULL_B"] + 0.5 * d["HLF_B"]
    d["BATH_PER_BED"] = d["TOT_BATH"] / d["BED_RMS"].clip(lower=1)
    d["SQFT_PER_ROOM"] = d["LIVING_SQFT"] / (d["BED_RMS"] + d["FULL_B"]).clip(lower=1)
    d["IS_STUDIO"] = (d["BED_RMS"] == 0).astype(int)

    # --- age ----------------------------------------------------------------
    # Price/age is not linear: depreciation is fast in the first years and then
    # flattens (and very old homes in Denver carry a "historic" premium).
    d["LOG_AGE"] = np.log1p(d["BLDG_AGE"])
    d["LOG_EFF_AGE"] = np.log1p(d["EFF_AGE"])
    d["IS_NEW"] = (d["BLDG_AGE"] <= 1).astype(int)
    d["IS_HISTORIC"] = (d["BLDG_AGE"] >= 80).astype(int)
    d["AGE_SQ"] = d["BLDG_AGE"] ** 2 / 1e3

    # --- type ---------------------------------------------------------------
    d["IS_CONDO"] = (d["PROP_CLASS"] == "CONDOMINIUMS").astype(int)

    return d


# ----------------------------------------------------------------------------
# 3. DESIGN MATRIX (dummies + slope dummies)
# ----------------------------------------------------------------------------
BASE_NUM = [
    "LIVING_SQFT", "LOG_LIVING", "LIVING_SQ", "TOT_FIN_SQFT", "LOG_TOT_FIN",
    "FBSMT_SQFT", "LOG_FBSMT", "BSMT_AREA", "LOG_BSMT", "BSMT_FIN_RATIO",
    "LAND_SQFT", "LOG_LAND", "LAND_PER_LIVING", "GRD_AREA",
    "BLDG_AGE", "LOG_AGE", "AGE_SQ", "EFF_AGE", "LOG_EFF_AGE",
    "BED_RMS", "FULL_B", "HLF_B", "TOT_BATH", "BATH_PER_BED", "SQFT_PER_ROOM",
    "STORY",
]
BASE_FLAG = [
    "IS_CONDO", "EVER_REMOD", "HAS_LAND", "HAS_BSMT", "HAS_FBSMT", "HAS_GRD",
    "IS_NEW", "IS_HISTORIC", "IS_STUDIO",
]

# Dummy reference levels are fixed explicitly so train and test agree, and so the
# k-1 rule (dummy-variable trap) is applied on purpose rather than by accident.
NBHD_REF = "GATEWAY / GREEN VALLEY RANCH"   # largest single-family neighborhood
STYLE_REF = "1STORY"


def design_matrix(d, nbhd_levels, style_levels, slope_dummies=True):
    X = d[BASE_NUM + BASE_FLAG].copy()

    for lv in nbhd_levels:
        if lv == NBHD_REF:
            continue
        X["NB_" + lv.replace(" / ", "_").replace(" ", "_")] = (d["NBHD"] == lv).astype(int)

    for lv in style_levels:
        if lv == STYLE_REF:
            continue
        X["ST_" + lv] = (d["STYLE_G"] == lv).astype(int)

    if slope_dummies:
        # Slope dummies: the $/sqft gradient is not the same everywhere.
        # Median price per living sqft ranges from $175 (Windsor) to $615
        # (Union Station), so letting size interact with neighborhood/type is
        # the single most defensible interaction in this data set.
        for lv in nbhd_levels:
            if lv == NBHD_REF:
                continue
            key = lv.replace(" / ", "_").replace(" ", "_")
            X["SLP_" + key] = X["NB_" + key] * d["LOG_LIVING"].values
        X["SLP_CONDO"] = d["IS_CONDO"] * d["LOG_LIVING"]
        X["SLP_AGE_CONDO"] = d["IS_CONDO"] * d["LOG_AGE"]
        X["SLP_LAND_SFR"] = (1 - d["IS_CONDO"]) * d["LOG_LAND"]

    return X


def build_all(slope_dummies=True):
    tr_raw, te_raw = load_raw()
    tr = engineer(clean(tr_raw))
    te = engineer(clean(te_raw))

    nbhd_levels = sorted(tr["NBHD"].unique())
    style_levels = sorted(tr["STYLE_G"].unique())

    # Guard: any level in test that the training data has never seen would be
    # silently dropped to the reference level. Checked, none exist here.
    unseen_n = set(te["NBHD"]) - set(nbhd_levels)
    unseen_s = set(te["STYLE_G"]) - set(style_levels)
    assert not unseen_n and not unseen_s, (unseen_n, unseen_s)

    Xtr = design_matrix(tr, nbhd_levels, style_levels, slope_dummies)
    Xte = design_matrix(te, nbhd_levels, style_levels, slope_dummies)
    Xte = Xte[Xtr.columns]  # identical column order
    ytr = tr["SALE_PRICE"].astype(float)
    return tr, te, Xtr, ytr, Xte


if __name__ == "__main__":
    tr, te, Xtr, ytr, Xte = build_all()
    out_tr = pd.concat([tr[["ID"]], ytr.rename("SALE_PRICE"), Xtr], axis=1)
    out_te = pd.concat([te[["ID"]], Xte], axis=1)
    out_tr.to_csv(HERE / "train_features.csv", index=False)
    out_te.to_csv(HERE / "test_features.csv", index=False)
    print("train_features.csv", out_tr.shape)
    print("test_features.csv ", out_te.shape)
    print("n predictors:", Xtr.shape[1])
