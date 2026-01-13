#!/usr/bin/env python3
"""
RF vs GeoGAM with (Spatial or Random) K-fold Cross-Validation

This script implements a "geographical GAM" (GeoGAM): a GAM that includes a
2D spatial smooth over coordinates (lon/lat) to absorb broad-scale spatial structure
and reduce residual spatial autocorrelation.

Key ideas
---------
- RF baseline trained on TRAIN folds only
- GeoGAM trained on TRAIN folds only:
    y ~ f(features) + te(lon, lat)
  where te(lon,lat) is a tensor-product smooth capturing spatial variation.
- Evaluate on TEST folds:
    R2, MAE, Moran's I on residuals (TEST only, using KNN weights)
- Exports per-fold predictions/residuals and validation points (GPKG) for QGIS.

Outputs (in OUTPUT_DIR)
-----------------------
- cv_predictions_residuals.csv
- cv_validation_points.gpkg
- cv_fold_metrics.csv
- cv_summary.txt
- residual_hist_{rf,geogam}_fold1_test.png
- obs_vs_pred_{rf,geogam}_fold1_test.png
- residuals_map_geogam_fold1_test.png
- rf_feature_importance_fold1.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold

import geopandas as gpd
from shapely.geometry import Point

# ------------------- CONFIG -------------------
METRICS_CSV     = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/platanus x acerifolia/ndvi_metrics_clean.csv'
TARGET_COL      = "auc_above_base_full"
FEATURE_COLS    = [
    "height","poll_pm25_anmean","poll_pm10_anmean","poll_no2_anmean","poll_bc_anmean",
    "impervious_r10","impervious_r20","impervious_r50","impervious_r100",
    "temp_r100","temp_r200"
]
LON_COL         = "lon"
LAT_COL         = "lat"

# Tree layer used to fetch lon/lat by ID
TREE_LAYER_PATH = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/tree layers/platanus_x_acerifolia.shp'
TREE_LAYER_NAME = None
CSV_ID_COL      = "tree_id"
LAYER_ID_COL    = "crown_id"

OUTPUT_DIR      = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/spatial regression/platanus x acerifolia'

# CV
N_FOLDS         = 10
USE_SPATIAL_CV  = False     # set True to use spatial folds via clustering
N_CLUSTERS      = 60
RANDOM_STATE    = 42

# RF baseline
N_ESTIMATORS_RF  = 500
MAX_FEATURES_RF  = "sqrt"
N_JOBS_RF        = -1

# GeoGAM settings (pyGAM)
# GeoGAM = linear terms for predictors + tensor-product smooth te(lon,lat)
# You can increase N_SPLINES_SPATIAL if you want a more flexible spatial field.
N_SPLINES_SPATIAL = 25          # per dimension in te(lon,lat)
GAM_GRIDSEARCH    = True        # gridsearch lambda on TRAIN folds
LAM_GRID          = np.logspace(-3, 3, 9)  # try fewer/more as needed
# ------------------------------------------------


def ensure_packages():
    """
    Required:
      pip install pygam esda==2.5 libpysal geopandas scikit-learn matplotlib pandas numpy shapely
    """
    try:
        import pygam  # noqa: F401
        from esda.moran import Moran  # noqa: F401
        import libpysal  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing packages. Install with:\n"
            "  pip install pygam esda==2.5 libpysal geopandas scikit-learn matplotlib pandas numpy shapely"
        ) from e

def lam_to_str(lam):
    """Readable representation of pyGAM lam (can be nested/ragged)."""
    try:
        if isinstance(lam, (list, tuple)):
            parts = []
            for x in lam:
                if isinstance(x, (list, tuple, np.ndarray)):
                    arr = np.asarray(x, dtype=float).ravel()
                    parts.append(f"[{arr.min():.2e}..{arr.max():.2e}]")
                else:
                    parts.append(f"{float(x):.2e}")
            return ";".join(parts)
        if isinstance(lam, np.ndarray):
            arr = np.asarray(lam, dtype=float).ravel()
            return f"[{arr.min():.2e}..{arr.max():.2e}]"
        return f"{float(lam):.2e}"
    except Exception:
        return str(lam)


def lam_to_mean(lam):
    """Flatten numeric content of pyGAM lam (ragged allowed) and return mean, else NaN."""
    vals = []

    def _collect(x):
        if x is None:
            return
        if isinstance(x, (list, tuple)):
            for y in x:
                _collect(y)
        elif isinstance(x, np.ndarray):
            arr = np.asarray(x, dtype=float).ravel()
            for v in arr:
                if np.isfinite(v):
                    vals.append(float(v))
        else:
            try:
                v = float(x)
                if np.isfinite(v):
                    vals.append(v)
            except Exception:
                pass

    _collect(lam)
    return float(np.mean(vals)) if vals else float("nan")

# --------- Join lon/lat from layer by ID ---------
def add_coords_by_id(
    df: pd.DataFrame,
    layer_path: str,
    layer_name: str | None,
    csv_id_col: str,
    layer_id_col: str,
    to_crs: str = "EPSG:4326",
    loncol: str = "lon",
    latcol: str = "lat",
    strict: bool = True
) -> pd.DataFrame:
    if csv_id_col not in df.columns:
        raise ValueError(f"CSV is missing id column '{csv_id_col}'.")
    gdf = gpd.read_file(layer_path, layer=layer_name) if layer_name else gpd.read_file(layer_path)
    if layer_id_col not in gdf.columns:
        raise ValueError(f"Tree layer is missing id column '{layer_id_col}'.")
    if gdf.crs is None:
        raise ValueError("Tree layer CRS is undefined; please define or reproject it first.")

    if "MultiPoint" in gdf.geometry.geom_type.unique():
        gdf = gdf.explode(index_parts=False)

    def _as_point(geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == "Point":
            return geom
        if geom.geom_type == "MultiPoint":
            return list(geom.geoms)[0] if len(geom.geoms) else None
        return geom.representative_point()

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf["__pt"] = gdf.geometry.apply(_as_point)
    gdf = gdf[gdf["__pt"].notnull()].set_geometry("__pt")

    if to_crs:
        gdf = gdf.to_crs(to_crs)

    gdf = gdf.drop_duplicates(subset=[layer_id_col])
    coords = pd.DataFrame({
        layer_id_col: gdf[layer_id_col].values,
        loncol: gdf.geometry.x.values,
        latcol: gdf.geometry.y.values,
    }).rename(columns={layer_id_col: csv_id_col})

    out = df.merge(coords, on=csv_id_col, how="left")
    missing = out[loncol].isna() | out[latcol].isna()
    if strict and int(missing.sum()) > 0:
        some = out.loc[missing, csv_id_col].dropna().astype(str).unique()[:10]
        raise ValueError(f"{int(missing.sum())} rows could not be matched by ID. Examples: {some}")
    return out


# --------- Build spatial folds via clusters ---------
def make_spatial_folds(coords_xy: np.ndarray, n_folds=5, n_clusters=60, random_state=42):
    """
    Cluster points and assign clusters to folds to reduce spatial leakage.
    """
    n = coords_xy.shape[0]
    mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=2048,
        n_init="auto"
    )
    labels = mbk.fit_predict(coords_xy)

    clusters = [np.where(labels == c)[0] for c in np.unique(labels)]
    clusters.sort(key=lambda idx: len(idx), reverse=True)

    target = int(np.ceil(n / n_folds))
    fold_tests = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for idx in clusters:
        j = int(np.argmin(fold_sizes))
        fold_tests[j].append(idx)
        fold_sizes[j] += len(idx)

    folds = []
    for j in range(n_folds):
        test_idx = np.concatenate(fold_tests[j]) if len(fold_tests[j]) else np.array([], dtype=int)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        folds.append((train_idx, test_idx))
    return folds


def make_random_folds(n, n_folds=10, random_state=42, shuffle=True):
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    return [(tr, te) for tr, te in kf.split(np.arange(n))]


# --------- Plot helper ---------
def _obs_pred(y_true, y_hat, name, fname, out_dir, target_label):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_hat  = np.asarray(y_hat,  dtype=float).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_hat)
    y_true, y_hat = y_true[m], y_hat[m]
    if y_true.size == 0:
        print(f"[warn] No finite pairs to plot for {name}.")
        return
    lo = min(y_true.min(), y_hat.min())
    hi = max(y_true.max(), y_hat.max())
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_true, y_hat, s=16, alpha=0.8)
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel(f"Observed {target_label}")
    plt.ylabel(f"Predicted {target_label} ({name})")
    plt.title(f"Observed vs Predicted ({name}) — TEST")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=160)
    plt.close()


def main():
    ensure_packages()

    from pygam import LinearGAM, l, te
    from esda.moran import Moran
    import libpysal as lps

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load & check
    df = pd.read_csv(METRICS_CSV)
    need = [TARGET_COL] + FEATURE_COLS + [CSV_ID_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # 2) Join lon/lat by ID
    df = add_coords_by_id(
        df=df,
        layer_path=TREE_LAYER_PATH,
        layer_name=TREE_LAYER_NAME,
        csv_id_col=CSV_ID_COL,
        layer_id_col=LAYER_ID_COL,
        to_crs="EPSG:4326",
        loncol=LON_COL,
        latcol=LAT_COL,
        strict=True
    )

    # 3) Filter rows
    cols_needed = [TARGET_COL, LON_COL, LAT_COL] + FEATURE_COLS
    df = df.dropna(subset=cols_needed).reset_index(drop=True)
    df = df[df[TARGET_COL] >= 0]
    if df.empty:
        raise ValueError("No valid rows after dropping NaNs/invalid values.")

    # 4) Matrices
    y_all = df[TARGET_COL].astype(float).values
    X_feat = df[FEATURE_COLS].astype(float).values
    coords = df[[LON_COL, LAT_COL]].astype(float).values

    # GeoGAM design matrix: [features..., lon, lat]
    X_gam = np.hstack([X_feat, coords])

    n_all = len(df)
    p_feat = len(FEATURE_COLS)
    lon_idx = p_feat
    lat_idx = p_feat + 1

    # 5) Folds
    if USE_SPATIAL_CV:
        folds = make_spatial_folds(coords, n_folds=N_FOLDS, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    else:
        folds = make_random_folds(n_all, n_folds=N_FOLDS, random_state=RANDOM_STATE)

    all_preds = []
    fold_metrics = []

    for fold_id, (tr_idx, te_idx) in enumerate(folds, start=1):
        if te_idx.size == 0 or tr_idx.size == 0:
            print(f"[warn] Fold {fold_id} is empty; skipping.")
            continue

        # Split
        Xtr_feat, Xte_feat = X_feat[tr_idx], X_feat[te_idx]
        ytr, yte = y_all[tr_idx], y_all[te_idx]
        Ctr, Cte = coords[tr_idx], coords[te_idx]

        Xtr_gam, Xte_gam = X_gam[tr_idx], X_gam[te_idx]

        # ---- RF baseline ----
        rf = RandomForestRegressor(
            n_estimators=N_ESTIMATORS_RF,
            max_features=MAX_FEATURES_RF,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_RF
        )
        rf.fit(Xtr_feat, ytr)
        yhat_rf_tr = rf.predict(Xtr_feat)
        yhat_rf_te = rf.predict(Xte_feat)

        resid_rf_te = yte - yhat_rf_te
        w_rf_te = lps.weights.KNN.from_array(Cte, k=12)
        w_rf_te.transform = "r"
        moran_rf_te = Moran(resid_rf_te, w_rf_te)

        r2_rf_tr  = r2_score(ytr, yhat_rf_tr)
        mae_rf_tr = mean_absolute_error(ytr, yhat_rf_tr)
        r2_rf_te  = r2_score(yte, yhat_rf_te)
        mae_rf_te = mean_absolute_error(yte, yhat_rf_te)

        # ---- GeoGAM: linear predictor terms + spatial te(lon,lat) ----
        # model: y ~ sum_i l(i) + te(lon, lat)
        # (You can replace some l(i) with s(i) smooths if you want nonlinearity in covariates.)
        terms = None
        for j in range(p_feat):
            terms = l(j) if terms is None else (terms + l(j))
        terms = terms + te(lon_idx, lat_idx, n_splines=[N_SPLINES_SPATIAL, N_SPLINES_SPATIAL])

        gam = LinearGAM(terms)

        if GAM_GRIDSEARCH:
            # Gridsearch only on TRAIN folds
            # Note: pygam gridsearch can be slow if you make the grid huge.
            gam.gridsearch(Xtr_gam, ytr, lam=LAM_GRID, progress=False)
        else:
            gam.fit(Xtr_gam, ytr)

        yhat_gam_tr = gam.predict(Xtr_gam)
        yhat_gam_te = gam.predict(Xte_gam)

        resid_gam_te = yte - yhat_gam_te
        w_gam_te = lps.weights.KNN.from_array(Cte, k=12)
        w_gam_te.transform = "r"
        moran_gam_te = Moran(resid_gam_te, w_gam_te)

        r2_gam_tr  = r2_score(ytr, yhat_gam_tr)
        mae_gam_tr = mean_absolute_error(ytr, yhat_gam_tr)
        r2_gam_te  = r2_score(yte, yhat_gam_te)
        mae_gam_te = mean_absolute_error(yte, yhat_gam_te)

        lam_val = getattr(gam, "lam", None)

        # Fold metrics
        fold_metrics.append({
            "fold": fold_id,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),

            "rf_r2_train": r2_rf_tr,
            "rf_mae_train": mae_rf_tr,
            "rf_r2_test": r2_rf_te,
            "rf_mae_test": mae_rf_te,
            "rf_moranI_test": moran_rf_te.I,
            "rf_moranP_test": moran_rf_te.p_norm,

            "geogam_r2_train": r2_gam_tr,
            "geogam_mae_train": mae_gam_tr,
            "geogam_r2_test": r2_gam_te,
            "geogam_mae_test": mae_gam_te,
            "geogam_moranI_test": moran_gam_te.I,
            "geogam_moranP_test": moran_gam_te.p_norm,

            "geogam_n_splines_spatial": N_SPLINES_SPATIAL,
            "geogam_lam_mean": lam_to_mean(lam_val),
            "geogam_lam_str": lam_to_str(lam_val),
        })


        # Predictions table (TEST only)
        fold_df = df.iloc[te_idx].copy()
        fold_df["fold"] = fold_id
        fold_df["split"] = "test"
        fold_df["pred_rf"] = yhat_rf_te
        fold_df["resid_rf"] = resid_rf_te
        fold_df["pred_geogam"] = yhat_gam_te
        fold_df["resid_geogam"] = resid_gam_te
        all_preds.append(fold_df)

        # Plots for fold 1
        if fold_id == 1:
            # Residual histograms
            plt.figure(figsize=(6, 4))
            plt.hist(resid_gam_te, bins=30, alpha=0.9)
            plt.axvline(0, color="k", lw=1)
            plt.title(
                f"GeoGAM TEST residuals (fold {fold_id})\n"
                f"R2={r2_gam_te:.3f}, MAE={mae_gam_te:.3f}, Moran's I={moran_gam_te.I:.3f} (p={moran_gam_te.p_norm:.4f})"
            )
            plt.xlabel("Residual")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"residual_hist_geogam_fold{fold_id}_test.png"), dpi=160)
            plt.close()

            plt.figure(figsize=(6, 4))
            plt.hist(resid_rf_te, bins=30, alpha=0.9)
            plt.axvline(0, color="k", lw=1)
            plt.title(
                f"RF TEST residuals (fold {fold_id})\n"
                f"R2={r2_rf_te:.3f}, MAE={mae_rf_te:.3f}, Moran's I={moran_rf_te.I:.3f} (p={moran_rf_te.p_norm:.4f})"
            )
            plt.xlabel("Residual")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"residual_hist_rf_fold{fold_id}_test.png"), dpi=160)
            plt.close()

            # Obs vs Pred
            _obs_pred(yte, yhat_gam_te, "GeoGAM", f"obs_vs_pred_geogam_fold{fold_id}_test.png", OUTPUT_DIR, TARGET_COL)
            _obs_pred(yte, yhat_rf_te, "RF", f"obs_vs_pred_rf_fold{fold_id}_test.png", OUTPUT_DIR, TARGET_COL)

            # Residual map (GeoGAM)
            plt.figure(figsize=(6, 5))
            sc = plt.scatter(fold_df[LON_COL], fold_df[LAT_COL], c=fold_df["resid_geogam"],
                             s=18, cmap="RdBu", alpha=0.9)
            plt.colorbar(sc, label="Residual (GeoGAM, TEST)")
            plt.title(f"Residuals map (GeoGAM) — TEST fold {fold_id}")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"residuals_map_geogam_fold{fold_id}_test.png"), dpi=160)
            plt.close()

            # RF feature importance
            if hasattr(rf, "feature_importances_"):
                imp = np.array(rf.feature_importances_)
                order = np.argsort(imp)[::-1]
                plt.figure(figsize=(6, 4))
                plt.bar(range(len(order)), imp[order])
                plt.xticks(range(len(order)), [FEATURE_COLS[i] for i in order], rotation=45, ha="right")
                plt.ylabel("Importance")
                plt.title(f"RF feature importance (TRAIN — fold {fold_id})")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"rf_feature_importance_fold{fold_id}.png"), dpi=160)
                plt.close()

    # Save predictions across folds
    if all_preds:
        preds = pd.concat(all_preds, ignore_index=True)
        preds_csv_path = os.path.join(OUTPUT_DIR, "cv_predictions_residuals.csv")
        preds.to_csv(preds_csv_path, index=False)
        print(f"[OK] Saved per-fold validation predictions/residuals: {preds_csv_path}")

        # Export validation points (GPKG for QGIS)
        gdf_val = gpd.GeoDataFrame(
            preds.copy(),
            geometry=gpd.points_from_xy(preds[LON_COL], preds[LAT_COL]),
            crs="EPSG:4326"
        )
        gpkg_path = os.path.join(OUTPUT_DIR, "cv_validation_points.gpkg")
        gdf_val.to_file(gpkg_path, layer="validation_points", driver="GPKG")
        print(f"[OK] Saved validation points vector layer: {gpkg_path} (layer='validation_points')")
    else:
        print("[WARN] No predictions to save.")
        preds = None

    # Save fold metrics
    dfm = pd.DataFrame(fold_metrics)
    fold_metrics_path = os.path.join(OUTPUT_DIR, "cv_fold_metrics.csv")
    dfm.to_csv(fold_metrics_path, index=False)
    print(f"[OK] Saved fold metrics: {fold_metrics_path}")

    # Summary
    summary_path = os.path.join(OUTPUT_DIR, "cv_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"K-fold CV (K={N_FOLDS})\n")
        f.write(f"CV type: {'SPATIAL' if USE_SPATIAL_CV else 'RANDOM'}\n")
        f.write(f"Target: {TARGET_COL}\n")
        f.write(f"Features: {FEATURE_COLS}\n")
        f.write(f"Samples: {n_all}\n\n")

        def mean_std(col):
            return float(dfm[col].mean()), float(dfm[col].std())

        for model in ["rf", "geogam"]:
            r2m, r2s = mean_std(f"{model}_r2_test")
            maem, maes = mean_std(f"{model}_mae_test")
            mim, mis = mean_std(f"{model}_moranI_test")
            f.write(
                f"{model.upper()} TEST: "
                f"R2={r2m:.4f}±{r2s:.4f}, "
                f"MAE={maem:.4f}±{maes:.4f}, "
                f"Moran's I={mim:.4f}±{mis:.4f}\n"
            )

        f.write("\nGeoGAM settings:\n")
        f.write(f"  spatial te(lon,lat) n_splines per dim: {N_SPLINES_SPATIAL}\n")
        f.write(f"  gridsearch lambda: {GAM_GRIDSEARCH}\n")
        if "geogam_lam_mean" in dfm.columns:
            f.write(f"  lam (mean across folds): {dfm['geogam_lam_mean'].mean():.6f}\n")

        if "geogam_lam_str" in dfm.columns and len(dfm) > 0:
            f.write(f"  lam (example fold): {dfm['geogam_lam_str'].iloc[0]}\n")

    print(f"[OK] Summary written: {summary_path}")
    print("[Done] K-fold CV — RF baseline + GeoGAM complete.")


if __name__ == "__main__":
    main()
