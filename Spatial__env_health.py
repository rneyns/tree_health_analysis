# pip install pandas geopandas scikit-learn shapely pyproj matplotlib numpy
# optional: pip install shapely[vectorized]

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
CSV_PATH = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/ndvi_metrics_clean.csv'        # your file
# If your CSV lacks coordinates, provide a point layer to fetch them:
POINTS_PATH = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/tree layers/Acer_platanoides.shp'  # any vector with Point geom & 'tree_id' field
POINTS_LAYER = None                           # or set to layer name; None = default
ID_FIELD = "tree_id"

# target(s) to model (choose one at a time)
TARGET = "auc_full"               # try also "sos_doy" or "slope_sos_peak"

# families of multi-scale predictors (edit to match your columns)
IMPERV_FAMS = [["impervious_r10","impervious_r20","impervious_r50","impervious_r50"]]
TEMP_FAMS   = [["temp_r100","temp_r200"]]
PM_COL      = "pm25"               # will be residualized vs temp_r10 (edit if needed)

# block/grid size for spatial CV (meters)
BLOCK_M = 1000

# optional quality filter for targets (example: plausible SOS range in Belgium)
FILTER_SOS = True
SOS_MIN, SOS_MAX = 80, 140

RANDOM_STATE = 42
# -------------------------------------------

def load_data_with_coords(csv_path, points_path=None, points_layer=None, id_field="tree_id"):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    # If CSV already has coords, use them
    if {"x","y"}.issubset(df.columns):
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:4326")
        gdf = gdf.to_crs(31370)  # Belgian Lambert 72
        gdf["x_m"] = gdf.geometry.x
        gdf["y_m"] = gdf.geometry.y
        return gdf

    if points_path is None:
        raise RuntimeError("No coordinates available. Provide x,y in CSV or set POINTS_PATH.")

    pts = gpd.read_file(points_path, layer=points_layer)
    if pts.crs is None:
        raise RuntimeError("Point layer CRS is undefined.")
    # Keep only id + geometry
    keep = [id_field, "geometry"]
    pts = pts[keep].drop_duplicates(subset=[id_field])
    # Join
    m = df.merge(pts[[id_field, "geometry"]], on=id_field, how="left")
    gdf = gpd.GeoDataFrame(m, geometry="geometry", crs=pts.crs)
    gdf = gdf.to_crs(31370)  # meters
    gdf["x_m"] = gdf.geometry.x
    gdf["y_m"] = gdf.geometry.y
    return gdf

def make_spatial_blocks(gdf, block_m=1000):
    bx = (gdf["x_m"] // block_m).astype(int)
    by = (gdf["y_m"] // block_m).astype(int)
    return (bx.astype(str) + "_" + by.astype(str)).astype("category")

def best_by_corr(X, y, colset):
    cols = [c for c in colset if c in X.columns and np.isfinite(X[c]).any()]
    if not cols:
        return None
    # use absolute Pearson corr on non-na rows
    best_c, best_r = None, -np.inf
    for c in cols:
        m = np.isfinite(X[c]) & np.isfinite(y)
        if m.sum() < 10:
            continue
        r = np.corrcoef(X.loc[m, c], y.loc[m])[0,1]
        r = np.abs(r) if np.isfinite(r) else 0
        if r > best_r:
            best_r, best_c = r, c
    return best_c

def residualize_pm_on_temp(df, pm_col, temp_col, groups):
    """Fold-safe residualization: fit on train, apply to test within each group fold."""
    df = df.copy()
    df["pm25_res"] = np.nan
    gkf = GroupKFold(n_splits=5)
    idx = np.arange(len(df))
    for tr, te in gkf.split(idx, groups=groups):
        tr, te = idx[tr], idx[te]
        m = df.loc[tr, [pm_col, temp_col]].dropna()
        if m.empty:
            continue
        lr = LinearRegression().fit(m[[temp_col]], m[pm_col])
        pred_tr = lr.predict(df.loc[tr, [temp_col]].values)
        pred_te = lr.predict(df.loc[te, [temp_col]].values)
        df.loc[tr, "pm25_res"] = df.loc[tr, pm_col] - pred_tr
        df.loc[te, "pm25_res"] = df.loc[te, pm_col] - pred_te
    return df

def run_elasticnet_spatial_cv(gdf, target, groups, feature_families, extra_features=None, title="model"):
    y = gdf[target].astype(float)
    X = gdf[sorted(set([c for fam in feature_families for c in fam if c in gdf.columns]))].copy()
    if extra_features:
        for c in extra_features:
            if c in gdf.columns:
                X[c] = gdf[c]

    # choose best column per family
    chosen = []
    for fam in feature_families:
        col = best_by_corr(X, y, fam)
        if col: chosen.append(col)
    if extra_features:
        chosen += [c for c in extra_features if c in X.columns]

    X = X[chosen].copy()
    # drop rows with NA in y or all features
    m = np.isfinite(y)
    for c in X.columns:
        m &= np.isfinite(X[c])
    X, y, groups = X.loc[m], y.loc[m], groups[m]

    scaler = StandardScaler()
    gkf = GroupKFold(n_splits=5)
    r2s, maes, importances = [], [], []

    # Collect out-of-fold predictions to report overall performance
    yhat_oof = pd.Series(index=y.index, dtype=float)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        en = ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=5, n_alphas=100, max_iter=5000, random_state=RANDOM_STATE)
        en.fit(Xtr_s, ytr)
        yhat = en.predict(Xte_s)
        yhat_oof.iloc[te] = yhat

        r2s.append(r2_score(yte, yhat))
        maes.append(mean_absolute_error(yte, yhat))

        # permutation importance on the held-out fold
        pi = permutation_importance(en, Xte_s, yte, n_repeats=50, random_state=RANDOM_STATE)
        imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
        importances.append(imp)

    print(f"\n[{title}] Spatial 5-fold (block) CV:")
    print(f"R²  mean±sd: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
    print(f"MAE mean±sd: {np.mean(maes):.4f} ± {np.std(maes):.4f}")

    # aggregate permutation importance across folds
    imp_df = pd.concat(importances, axis=1).fillna(0)
    imp_df.columns = [f"fold{i+1}" for i in range(len(importances))]
    imp_df["mean"] = imp_df.mean(axis=1)
    print("\nPermutation importance (mean across folds):")
    print(imp_df["mean"].sort_values(ascending=False))

    # PDPs for top 3 features (using all data, indicative only)
    top3 = imp_df["mean"].sort_values(ascending=False).head(3).index.tolist()
    if top3:
        fig, ax = plt.subplots(1, len(top3), figsize=(4*len(top3), 3), constrained_layout=True)
        if len(top3) == 1: ax = [ax]
        Xs = scaler.fit_transform(X)  # re-fit on all data for PDP
        for i, feat in enumerate(top3):
            try:
                PartialDependenceDisplay.from_estimator(
                    ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=5, n_alphas=50, random_state=RANDOM_STATE).fit(Xs, y),
                    Xs, features=[list(X.columns).index(feat)], feature_names=list(X.columns), ax=ax[i]
                )
                ax[i].set_title(f"PDP: {feat}")
            except Exception:
                ax[i].set_title(f"PDP: {feat} (failed)")
        plt.show()

    return {
        "chosen_features": chosen,
        "r2s": r2s,
        "maes": maes,
        "perm_importance": imp_df.sort_values("mean", ascending=False),
        "oof_pred": yhat_oof
    }

def main():
    gdf = load_data_with_coords(CSV_PATH, POINTS_PATH, POINTS_LAYER, ID_FIELD)

    # (optional) filter obviously implausible SOS values
    if FILTER_SOS and "sos_doy" in gdf.columns:
        m = (gdf["sos_doy"].between(SOS_MIN, SOS_MAX)) | (~np.isfinite(gdf["sos_doy"]))
        before = len(gdf)
        gdf = gdf[m].copy()
        print(f"Filtered SOS to {SOS_MIN}-{SOS_MAX}: kept {len(gdf)}/{before}")

    # build spatial blocks
    gdf["block_id"] = make_spatial_blocks(gdf, BLOCK_M)

    # residualize PM2.5 on temperature (use r10 by default; change if needed)
    temp_anchor = None
    for cand in ["temp_r100","temp_r200"]:
        if cand in gdf.columns:
            temp_anchor = cand; break
    if temp_anchor and PM_COL in gdf.columns:
        gdf = residualize_pm_on_temp(gdf, PM_COL, temp_anchor, groups=gdf["block_id"])
        pm_feature = "pm25_res"
        print(f"Residualized {PM_COL} on {temp_anchor} → pm25_res")
    else:
        pm_feature = PM_COL

    # choose families to feed
    fams = []
    for fam in IMPERV_FAMS: fams.append([c for c in fam if c in gdf.columns])
    for fam in TEMP_FAMS:   fams.append([c for c in fam if c in gdf.columns])

    # extra features (single columns that should always go in)
    extra = [pm_feature] if pm_feature in gdf.columns else []

    # run model
    res = run_elasticnet_spatial_cv(
        gdf=gdf,
        target=TARGET,
        groups=gdf["block_id"],
        feature_families=fams,
        extra_features=extra,
        title=f"ElasticNet on {TARGET}"
    )
    print("\nChosen features:", res["chosen_features"])

if __name__ == "__main__":
    main()
