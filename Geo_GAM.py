#!/usr/bin/env python3
"""
RF vs GAM (non-spatial) vs GeoGAM (spatial smooth) with K-fold Cross-Validation
+ Moran's I on TEST residuals
+ Semivariogram diagnostics (empirical variograms) for paper figures

What this script does
---------------------
1) Loads a per-tree metrics CSV (response + predictors + tree_id)
2) Joins lon/lat from a tree layer (shp/gpkg) by ID
3) Builds either RANDOM K-fold CV or SPATIAL K-fold CV (via coordinate clustering)
4) For each fold (TRAIN -> fit, TEST -> evaluate):
   - RF baseline
   - GAM (non-spatial) with non-linear smooths for chosen predictors
   - GeoGAM (spatial GAM) = same GAM + a 2D spatial smooth te(lon, lat)
   - Computes R2 and MAE on TRAIN and TEST
   - Computes Moran's I on TEST residuals for each model
5) Saves per-fold predictions/residuals and fold-level metrics
6) Exports TEST points as GPKG for QGIS
7) Creates paper-ready diagnostics for fold 1:
   - residual histograms
   - observed vs predicted
   - residual maps (GeoGAM)
   - RF feature importance (MDI) plot
   - semivariograms of TEST residuals (GAM vs GeoGAM) (+ optional raw y)

Dependencies
------------
pip install numpy pandas matplotlib scikit-learn geopandas shapely esda==2.5 libpysal pygam scikit-gstat

Notes
-----
- The GAMs here are implemented via pyGAM (LinearGAM), which supports smooth terms s()
  and tensor-product smooth te().
- Tree height is included as a *smooth control* term: we control for it but do not
  interpret it causally.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold

import geopandas as gpd

# ------------------- CONFIG -------------------
# Data
METRICS_CSV     = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/platanus x acerifolia/ndvi_metrics_clean.csv'
TREE_LAYER_PATH = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/tree layers/platanus_x_acerifolia.shp'
TREE_LAYER_NAME = None  # set if layer is inside a GPKG
CSV_ID_COL      = "tree_id"
LAYER_ID_COL    = "crown_id"

TARGET_COL      = "auc_above_base_full"
LON_COL         = "lon"
LAT_COL         = "lat"

# --- Choose predictors and how they enter the GAM ---
# You said: non-linear for height + impervious + temperature; linear for pollution
HEIGHT_COL      = "height"

# Pick ONE impervious metric to avoid collinearity in GAM smooths
# (RF can still use multiple, but GAM will be more stable with one.)
IMP_COL         = "impervious_r50"

# Pick ONE temperature metric
TEMP_COL        = "temp_r200"

# Pollution terms (linear)
POLL_COLS       = ["poll_pm25_anmean", "poll_pm10_anmean", "poll_no2_anmean", "poll_bc_anmean"]

# RF uses a broader set (can be same as GAM or larger)
FEATURE_COLS_RF = [
    HEIGHT_COL,
    *POLL_COLS,
    "impervious_r10","impervious_r20","impervious_r50","impervious_r100",
    "temp_r100","temp_r200"
]

OUTPUT_DIR      = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/spatial regression/platanus x acerifolia'

# CV
N_FOLDS         = 10
USE_SPATIAL_CV  = False     # True => spatially blocked folds via clustering
N_CLUSTERS      = 60
RANDOM_STATE    = 42

# RF baseline
N_ESTIMATORS_RF  = 500
MAX_FEATURES_RF  = "sqrt"
N_JOBS_RF        = -1

# GAM / GeoGAM
# k for univariate smooths: keep small to avoid overfitting and to be paper-friendly
K_HEIGHT     = 6
K_IMP        = 6
K_TEMP       = 6

# Spatial smooth flexibility
K_SPATIAL    = (25, 25)   # basis sizes for lon, lat in te()

# Lambda search (penalization strength). A small grid is usually enough for CV loops.
GAM_GRIDSEARCH = True
LAM_GRID = np.logspace(-3, 3, 9)

# Moran's I settings
MORAN_KNN_K  = 12

# Variogram settings (for paper figure, fold 1 only)
MAKE_VARIOGRAM_FOLD1 = True
VGRAM_N_LAGS = 15
VGRAM_MAXLAG = None   # set e.g. 300 (meters) if you use projected coords; for lon/lat keep None

# ------------------------------------------------


def ensure_packages():
    try:
        from pygam import LinearGAM, s, l, te  # noqa: F401
        from esda.moran import Moran  # noqa: F401
        import libpysal  # noqa: F401
        from skgstat import Variogram  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing packages. Install with:\n"
            "  pip install pygam esda==2.5 libpysal geopandas shapely "
            "scikit-learn matplotlib pandas numpy scikit-gstat"
        ) from e


# ---------------- Lambda helpers (avoid ragged lam issues) ----------------
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


def lam_to_str(lam):
    """Readable representation of pyGAM lam (ragged allowed)."""
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


# ---------------- Join lon/lat from layer by ID ----------------
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

    # Convert any crown polygons to points (representative points)
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

    # De-duplicate IDs so merge is 1:1
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


# ---------------- Fold builders ----------------
def make_spatial_folds(coords_xy: np.ndarray, n_folds=5, n_clusters=60, random_state=42):
    """
    Spatial folds via clustering (MiniBatchKMeans).
    Points are clustered and clusters are assigned to folds to reduce spatial leakage.
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

    fold_tests = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for idx in clusters:
        j = int(np.argmin(fold_sizes))
        fold_tests[j].append(idx)
        fold_sizes[j] += len(idx)

    folds = []
    for j in range(n_folds):
        test_idx = np.concatenate(fold_tests[j]) if fold_tests[j] else np.array([], dtype=int)
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        folds.append((train_idx, test_idx))
    return folds


def make_random_folds(n, n_folds=10, random_state=42, shuffle=True):
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    return [(tr, te) for tr, te in kf.split(np.arange(n))]


# ---------------- Plot helpers ----------------
def plot_obs_pred(y_true, y_hat, name, out_dir, fname, target_label):
    y_true = np.asarray(y_true, float).ravel()
    y_hat  = np.asarray(y_hat,  float).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_hat)
    y_true, y_hat = y_true[m], y_hat[m]
    if y_true.size == 0:
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


def plot_residual_hist(resid, title, out_dir, fname):
    resid = np.asarray(resid, float)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=30, alpha=0.9)
    plt.axvline(0, color="k", lw=1)
    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=160)
    plt.close()


def plot_residual_map(lon, lat, resid, title, out_dir, fname):
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    resid = np.asarray(resid, float)
    m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(resid)
    if m.sum() == 0:
        return
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(lon[m], lat[m], c=resid[m], s=18, cmap="RdBu", alpha=0.9)
    plt.colorbar(sc, label="Residual")
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=160)
    plt.close()


def plot_rf_importance(feature_names, importances, out_dir, fname, title):
    imp = np.asarray(importances, float)
    order = np.argsort(imp)[::-1]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(order)), imp[order])
    plt.xticks(range(len(order)), [feature_names[i] for i in order], rotation=45, ha="right")
    plt.ylabel("MDI importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=160)
    plt.close()


# ---------------- Variogram helper ----------------
def plot_semivariograms(coords_xy, series_dict, out_dir, fname, n_lags=15, maxlag=None):
    """
    series_dict: dict(label -> values)
    Uses scikit-gstat Variogram to compute empirical semivariogram.
    """
    from skgstat import Variogram

    # If you keep lon/lat, distances will be in "degrees". That's okay for shape comparisons,
    # but for a paper figure, a projected CRS (meters) is better.
    plt.figure(figsize=(6.5, 4.5))
    for label, vals in series_dict.items():
        vals = np.asarray(vals, float)
        m = np.isfinite(vals) & np.all(np.isfinite(coords_xy), axis=1)
        if m.sum() < 10:
            continue
        V = Variogram(
            coords_xy[m],
            vals[m],
            n_lags=n_lags,
            maxlag=maxlag,
            normalize=False
        )
        plt.plot(V.bins, V.experimental, "o-", label=label)

    plt.xlabel("Distance")
    plt.ylabel("Semivariance")
    plt.title("Empirical semivariograms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=160)
    plt.close()


# ---------------- Model builders ----------------
def build_gam_terms(poll_count: int, include_spatial: bool):
    """
    Construct pyGAM terms for:
      y ~ s(height) + s(imp) + s(temp) + linear(pollution...) [+ te(lon,lat)]
    The order of columns in X_gam must match:
      [height, imp, temp, poll..., lon, lat]
    """
    from pygam import s, l, te

    # indices in X_gam
    idx_height = 0
    idx_imp    = 1
    idx_temp   = 2
    idx_poll0  = 3
    idx_lon    = 3 + poll_count
    idx_lat    = idx_lon + 1

    terms = (
            s(idx_height, n_splines=K_HEIGHT) +
            s(idx_imp, n_splines=K_IMP) +
            s(idx_temp, n_splines=K_TEMP)
    )

    for j in range(poll_count):
        terms = terms + l(idx_poll0 + j)

    if include_spatial:
        terms = terms + te(idx_lon, idx_lat, n_splines=list(K_SPATIAL))
    return terms


def fit_gam(X_train, y_train, terms, do_gridsearch=True):
    """
    Fit a LinearGAM with optional lambda gridsearch on training data only.
    """
    from pygam import LinearGAM

    gam = LinearGAM(terms)
    if do_gridsearch:
        # gridsearch chooses smoothing penalty; keep grid modest inside CV loop
        gam.gridsearch(X_train, y_train, lam=LAM_GRID, progress=False)
    else:
        gam.fit(X_train, y_train)
    return gam


# ---------------- Main ----------------
def main():
    ensure_packages()

    from esda.moran import Moran
    import libpysal as lps

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load and validate columns
    df = pd.read_csv(METRICS_CSV)

    required = [CSV_ID_COL, TARGET_COL, HEIGHT_COL, IMP_COL, TEMP_COL, *POLL_COLS, *FEATURE_COLS_RF]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # 2) Join lon/lat
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

    # 3) Filter invalid rows
    needed = [TARGET_COL, LON_COL, LAT_COL, HEIGHT_COL, IMP_COL, TEMP_COL, *POLL_COLS, *FEATURE_COLS_RF]
    df = df.dropna(subset=needed).reset_index(drop=True)
    df = df[df[TARGET_COL] >= 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after filtering NaNs/invalid values.")

    # 4) Matrices
    y_all = df[TARGET_COL].astype(float).values
    coords_ll = df[[LON_COL, LAT_COL]].astype(float).values  # lon/lat

    # RF matrix
    X_rf = df[FEATURE_COLS_RF].astype(float).values

    # GAM matrix (ordered)
    gam_cols = [HEIGHT_COL, IMP_COL, TEMP_COL, *POLL_COLS, LON_COL, LAT_COL]
    X_gam_all = df[gam_cols].astype(float).values

    n = len(df)

    # 5) Folds
    if USE_SPATIAL_CV:
        folds = make_spatial_folds(coords_ll, n_folds=N_FOLDS, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    else:
        folds = make_random_folds(n, n_folds=N_FOLDS, random_state=RANDOM_STATE)

    # 6) Pre-build GAM term sets
    poll_count = len(POLL_COLS)
    terms_gam   = build_gam_terms(poll_count=poll_count, include_spatial=False)
    terms_geog  = build_gam_terms(poll_count=poll_count, include_spatial=True)

    all_preds = []
    fold_metrics = []

    for fold_id, (tr_idx, te_idx) in enumerate(folds, start=1):
        if te_idx.size == 0 or tr_idx.size == 0:
            print(f"[warn] Fold {fold_id} empty; skipping.")
            continue

        # Split
        Xrf_tr, Xrf_te = X_rf[tr_idx], X_rf[te_idx]
        Xg_tr,  Xg_te  = X_gam_all[tr_idx], X_gam_all[te_idx]
        y_tr, y_te     = y_all[tr_idx], y_all[te_idx]
        C_te           = coords_ll[te_idx]

        # ---------- RF ----------
        rf = RandomForestRegressor(
            n_estimators=N_ESTIMATORS_RF,
            max_features=MAX_FEATURES_RF,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_RF
        )
        rf.fit(Xrf_tr, y_tr)
        yhat_rf_tr = rf.predict(Xrf_tr)
        yhat_rf_te = rf.predict(Xrf_te)

        resid_rf_te = y_te - yhat_rf_te
        w_rf = lps.weights.KNN.from_array(C_te, k=MORAN_KNN_K)
        w_rf.transform = "r"
        moran_rf = Moran(resid_rf_te, w_rf)

        r2_rf_tr  = r2_score(y_tr, yhat_rf_tr)
        mae_rf_tr = mean_absolute_error(y_tr, yhat_rf_tr)
        r2_rf_te  = r2_score(y_te, yhat_rf_te)
        mae_rf_te = mean_absolute_error(y_te, yhat_rf_te)

        # ---------- GAM (non-spatial) ----------
        gam = fit_gam(Xg_tr, y_tr, terms_gam, do_gridsearch=GAM_GRIDSEARCH)
        yhat_gam_tr = gam.predict(Xg_tr)
        yhat_gam_te = gam.predict(Xg_te)

        resid_gam_te = y_te - yhat_gam_te
        w_gam = lps.weights.KNN.from_array(C_te, k=MORAN_KNN_K)
        w_gam.transform = "r"
        moran_gam = Moran(resid_gam_te, w_gam)

        r2_gam_tr  = r2_score(y_tr, yhat_gam_tr)
        mae_gam_tr = mean_absolute_error(y_tr, yhat_gam_tr)
        r2_gam_te  = r2_score(y_te, yhat_gam_te)
        mae_gam_te = mean_absolute_error(y_te, yhat_gam_te)

        lam_val_gam = getattr(gam, "lam", None)

        # ---------- GeoGAM (spatial smooth) ----------
        geogam = fit_gam(Xg_tr, y_tr, terms_geog, do_gridsearch=GAM_GRIDSEARCH)
        yhat_geog_tr = geogam.predict(Xg_tr)
        yhat_geog_te = geogam.predict(Xg_te)

        resid_geog_te = y_te - yhat_geog_te
        w_geog = lps.weights.KNN.from_array(C_te, k=MORAN_KNN_K)
        w_geog.transform = "r"
        moran_geog = Moran(resid_geog_te, w_geog)

        r2_geog_tr  = r2_score(y_tr, yhat_geog_tr)
        mae_geog_tr = mean_absolute_error(y_tr, yhat_geog_tr)
        r2_geog_te  = r2_score(y_te, yhat_geog_te)
        mae_geog_te = mean_absolute_error(y_te, yhat_geog_te)

        lam_val_geog = getattr(geogam, "lam", None)

        # ---------- Save fold metrics ----------
        fold_metrics.append({
            "fold": fold_id,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),

            "rf_r2_train": r2_rf_tr,
            "rf_mae_train": mae_rf_tr,
            "rf_r2_test": r2_rf_te,
            "rf_mae_test": mae_rf_te,
            "rf_moranI_test": moran_rf.I,
            "rf_moranP_test": moran_rf.p_norm,

            "gam_r2_train": r2_gam_tr,
            "gam_mae_train": mae_gam_tr,
            "gam_r2_test": r2_gam_te,
            "gam_mae_test": mae_gam_te,
            "gam_moranI_test": moran_gam.I,
            "gam_moranP_test": moran_gam.p_norm,
            "gam_lam_mean": lam_to_mean(lam_val_gam),
            "gam_lam_str": lam_to_str(lam_val_gam),

            "geogam_r2_train": r2_geog_tr,
            "geogam_mae_train": mae_geog_tr,
            "geogam_r2_test": r2_geog_te,
            "geogam_mae_test": mae_geog_te,
            "geogam_moranI_test": moran_geog.I,
            "geogam_moranP_test": moran_geog.p_norm,
            "geogam_lam_mean": lam_to_mean(lam_val_geog),
            "geogam_lam_str": lam_to_str(lam_val_geog),

            "gam_height_col": HEIGHT_COL,
            "gam_imp_col": IMP_COL,
            "gam_temp_col": TEMP_COL,
            "gam_poll_cols": ",".join(POLL_COLS),
            "geogam_spatial_k": f"{K_SPATIAL[0]}x{K_SPATIAL[1]}",
        })

        # ---------- Save per-tree TEST predictions ----------
        fold_df = df.iloc[te_idx].copy()
        fold_df["fold"] = fold_id
        fold_df["split"] = "test"

        fold_df["pred_rf"] = yhat_rf_te
        fold_df["resid_rf"] = resid_rf_te

        fold_df["pred_gam"] = yhat_gam_te
        fold_df["resid_gam"] = resid_gam_te

        fold_df["pred_geogam"] = yhat_geog_te
        fold_df["resid_geogam"] = resid_geog_te

        all_preds.append(fold_df)

        # ---------- Fold 1 plots for paper ----------
        if fold_id == 1:
            plot_residual_hist(
                resid_rf_te,
                f"RF TEST residuals (fold 1)\nR2={r2_rf_te:.3f}, MAE={mae_rf_te:.3f}, Moran's I={moran_rf.I:.3f} (p={moran_rf.p_norm:.4f})",
                OUTPUT_DIR, "residual_hist_rf_fold1_test.png"
            )
            plot_residual_hist(
                resid_gam_te,
                f"GAM (non-spatial) TEST residuals (fold 1)\nR2={r2_gam_te:.3f}, MAE={mae_gam_te:.3f}, Moran's I={moran_gam.I:.3f} (p={moran_gam.p_norm:.4f})",
                OUTPUT_DIR, "residual_hist_gam_fold1_test.png"
            )
            plot_residual_hist(
                resid_geog_te,
                f"GeoGAM TEST residuals (fold 1)\nR2={r2_geog_te:.3f}, MAE={mae_geog_te:.3f}, Moran's I={moran_geog.I:.3f} (p={moran_geog.p_norm:.4f})",
                OUTPUT_DIR, "residual_hist_geogam_fold1_test.png"
            )

            plot_obs_pred(y_te, yhat_rf_te, "RF", OUTPUT_DIR, "obs_vs_pred_rf_fold1_test.png", TARGET_COL)
            plot_obs_pred(y_te, yhat_gam_te, "GAM", OUTPUT_DIR, "obs_vs_pred_gam_fold1_test.png", TARGET_COL)
            plot_obs_pred(y_te, yhat_geog_te, "GeoGAM", OUTPUT_DIR, "obs_vs_pred_geogam_fold1_test.png", TARGET_COL)

            plot_residual_map(
                fold_df[LON_COL], fold_df[LAT_COL], fold_df["resid_geogam"],
                "Residuals map (GeoGAM) — TEST fold 1",
                OUTPUT_DIR, "residuals_map_geogam_fold1_test.png"
            )

            # RF feature importance (MDI)
            if hasattr(rf, "feature_importances_"):
                plot_rf_importance(
                    FEATURE_COLS_RF,
                    rf.feature_importances_,
                    OUTPUT_DIR,
                    "rf_feature_importance_fold1.png",
                    "RF feature importance (MDI) — TRAIN fold 1"
                )

            # Semivariograms (fold 1, TEST only)
            if MAKE_VARIOGRAM_FOLD1:
                # Note: coords_ll are lon/lat degrees. For a *paper*, consider reprojecting
                # coords to a projected CRS (meters) prior to variogram calculation.
                series = {
                    "Residuals GAM (non-spatial)": resid_gam_te,
                    "Residuals GeoGAM (spatial)": resid_geog_te,
                    "Raw y (TEST)": y_te
                }
                plot_semivariograms(
                    coords_xy=C_te,
                    series_dict=series,
                    out_dir=OUTPUT_DIR,
                    fname="semivariogram_fold1_test.png",
                    n_lags=VGRAM_N_LAGS,
                    maxlag=VGRAM_MAXLAG
                )

    # ---------- Save outputs ----------
    # Predictions across folds
    if all_preds:
        preds = pd.concat(all_preds, ignore_index=True)
        preds_csv = os.path.join(OUTPUT_DIR, "cv_predictions_residuals.csv")
        preds.to_csv(preds_csv, index=False)
        print(f"[OK] Saved predictions/residuals: {preds_csv}")

        # Export validation points to GPKG for QGIS
        gdf_val = gpd.GeoDataFrame(
            preds.copy(),
            geometry=gpd.points_from_xy(preds[LON_COL], preds[LAT_COL]),
            crs="EPSG:4326"
        )
        gpkg_path = os.path.join(OUTPUT_DIR, "cv_validation_points.gpkg")
        gdf_val.to_file(gpkg_path, layer="validation_points", driver="GPKG")
        print(f"[OK] Saved validation points: {gpkg_path} (layer='validation_points')")
    else:
        print("[WARN] No fold predictions generated.")

    # Fold metrics
    dfm = pd.DataFrame(fold_metrics)
    metrics_csv = os.path.join(OUTPUT_DIR, "cv_fold_metrics.csv")
    dfm.to_csv(metrics_csv, index=False)
    print(f"[OK] Saved fold metrics: {metrics_csv}")

    # Summary
    summary_path = os.path.join(OUTPUT_DIR, "cv_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"K-fold CV (K={N_FOLDS})\n")
        f.write(f"CV type: {'SPATIAL' if USE_SPATIAL_CV else 'RANDOM'}\n\n")
        f.write(f"Target: {TARGET_COL}\n")
        f.write(f"RF features: {FEATURE_COLS_RF}\n")
        f.write(f"GAM columns: height={HEIGHT_COL}, impervious={IMP_COL}, temp={TEMP_COL}, poll={POLL_COLS}\n")
        f.write(f"Spatial smooth: te(lon,lat) with k={K_SPATIAL}\n")
        f.write(f"Samples: {len(df)}\n\n")

        def mean_std(col):
            return float(dfm[col].mean()), float(dfm[col].std())

        for model in ["rf", "gam", "geogam"]:
            r2m, r2s = mean_std(f"{model}_r2_test")
            maem, maes = mean_std(f"{model}_mae_test")
            mim, mis = mean_std(f"{model}_moranI_test")
            f.write(
                f"{model.upper()} TEST: "
                f"R2={r2m:.4f}±{r2s:.4f}, "
                f"MAE={maem:.4f}±{maes:.4f}, "
                f"Moran's I={mim:.4f}±{mis:.4f}\n"
            )

        if "gam_lam_mean" in dfm.columns:
            f.write("\nLambda (penalty) summaries:\n")
            f.write(f"  GAM lam mean (across folds):    {dfm['gam_lam_mean'].mean():.6f}\n")
            f.write(f"  GeoGAM lam mean (across folds): {dfm['geogam_lam_mean'].mean():.6f}\n")
            if len(dfm) > 0:
                f.write(f"  Example GAM lam (fold 1):       {dfm['gam_lam_str'].iloc[0]}\n")
                f.write(f"  Example GeoGAM lam (fold 1):    {dfm['geogam_lam_str'].iloc[0]}\n")

    print(f"[OK] Summary written: {summary_path}")
    print("[Done] RF vs GAM vs GeoGAM CV complete.")


if __name__ == "__main__":
    # silence some benign pygam warnings in CV loops
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
