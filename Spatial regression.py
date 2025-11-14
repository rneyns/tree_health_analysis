#!/usr/bin/env python3
"""
GRF vs RF with Spatial K-fold Cross-Validation

- Joins lon/lat from tree layer by ID (CSV: tree_id, Layer: crown_id)
- Spatial K-fold CV via coordinate clustering (MiniBatchKMeans)
- RF baseline and GRF (RAM-aware) trained only on TRAIN folds
- Reports per-fold and aggregated TEST metrics + Moran's I for residuals

Outputs (in OUTPUT_DIR)
-----------------------
- cv_predictions_residuals.csv         (per tree per fold with split flag)
- cv_fold_metrics.csv                  (metrics per fold/model)
- cv_summary.txt                       (mean±std metrics + chosen GRF params ranges)
- residual_hist_{rf,grf}_fold1_test.png
- obs_vs_pred_{rf,grf}_fold1_test.png
- residuals_map_grf_fold1_test.png
- global_feature_importance_rf_fold1.png
- pdp_rf_top_features_fold1.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay
from sklearn.cluster import MiniBatchKMeans

# ------------------- CONFIG -------------------
METRICS_CSV     = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/ndvi_metrics_clean.csv'
TARGET_COL      = "auc_above_base_full"
FEATURE_COLS    = ["pm25","impervious_r10","impervious_r20","impervious_r50","impervious_r100","temp_r100","temp_r200"]
LON_COL         = "lon"
LAT_COL         = "lat"

# Tree layer used to fetch lon/lat by ID
TREE_LAYER_PATH = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/tree layers/platanus_x_acerifolia.shp'
TREE_LAYER_NAME = None
CSV_ID_COL      = "tree_id"
LAYER_ID_COL    = "crown_id"

OUTPUT_DIR      = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/spatial regression'

# Spatial K-fold
N_FOLDS         = 10        # number of spatial folds
N_CLUSTERS      = 60       # number of spatial clusters used to define folds
TEST_MIN_FRACT  = 0.15     # aim ~1/N_FOLDS; this is a lower bound per held cluster pack
RANDOM_STATE    = 42

# ---- GRF defaults (auto-scaled by RAM per fold) ----
N_ESTIMATORS_GRF_DEFAULT = 50
MAX_FEATURES_GRF         = None   # if None -> int(sqrt(p))
BAND_WIDTH_DEFAULT       = 40
TRAIN_WEIGHTED_DEFAULT   = True
PRED_WEIGHTED_DEFAULT    = True
BOOTSTRAP_DEFAULT        = False
RESAMPLED_DEFAULT        = True
LOCAL_WEIGHT             = 0.5

# RF baseline
N_ESTIMATORS_RF  = 500
MAX_FEATURES_RF  = "sqrt"
N_JOBS_RF        = -1



# ------------------------------------------------


def ensure_packages():
    try:
        from PyGRF import PyGRF  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing PyGRF. Install with:\n"
            "  pip install PyGRF esda==2.5 libpysal geopandas scikit-learn matplotlib pandas numpy"
        ) from e


# --------- RAM-aware helpers ---------
def get_available_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None

def choose_grf_params(avail_gb, p_features, n_train):
    n_estimators = N_ESTIMATORS_GRF_DEFAULT
    band_width   = BAND_WIDTH_DEFAULT
    train_w      = TRAIN_WEIGHTED_DEFAULT
    pred_w       = PRED_WEIGHTED_DEFAULT
    resampled    = RESAMPLED_DEFAULT

    if avail_gb is None:
        band_width, n_estimators = 80, 120
    else:
        if avail_gb >= 64:   band_width, n_estimators = 300, 400
        elif avail_gb >= 32: band_width, n_estimators = 220, 300
        elif avail_gb >= 16: band_width, n_estimators = 160, 200
        elif avail_gb >= 8:  band_width, n_estimators = 120, 150
        else:
            band_width, n_estimators = 80, 100
            resampled = False

        # guardrails relative to train size
        band_width   = int(max(40, min(band_width, max(60, int(0.12 * n_train)))))
        n_estimators = int(max(60, min(n_estimators, 600)))

    max_feats_int = max(1, int(np.sqrt(p_features))) if MAX_FEATURES_GRF is None else int(MAX_FEATURES_GRF)
    return {
        "n_estimators": n_estimators,
        "band_width": band_width,
        "max_features": max_feats_int,
        "train_weighted": train_w,
        "predict_weighted": pred_w,
        "bootstrap": BOOTSTRAP_DEFAULT,
        "resampled": resampled,
        "local_weight": LOCAL_WEIGHT
    }


# --------- Join lon/lat from layer by ID ---------
import geopandas as gpd
from shapely.geometry import Point

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
def make_spatial_folds(coords_xy: np.ndarray, n_folds=5, n_clusters=60, min_test_fract=0.15, random_state=42):
    """
    Returns a list of (train_idx, test_idx) tuples for spatial K-fold CV.
    We cluster points; then greedily pack clusters into each fold until target size.
    """
    n = coords_xy.shape[0]
    # cluster in lon/lat (for Brussels scale this is fine; for meters reproject beforehand)
    mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=2048, n_init="auto")
    labels = mbk.fit_predict(coords_xy)

    # group indices by cluster
    clusters = [np.where(labels == c)[0] for c in np.unique(labels)]
    # sort clusters by size (desc) for stable packing
    clusters.sort(key=lambda idx: len(idx), reverse=True)

    # target test size per fold
    target = int(np.ceil(n / n_folds))
    # greedy packing
    fold_tests = [[] for _ in range(n_folds)]
    fold_sizes = [0]*n_folds

    for idx in clusters:
        # put this cluster into the fold with smallest current size
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

from sklearn.model_selection import KFold

def make_random_folds(n, n_folds=10, random_state=42, shuffle=True):
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    folds = []
    for tr_idx, te_idx in kf.split(np.arange(n)):
        folds.append((tr_idx, te_idx))
    return folds


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
    plt.figure(figsize=(5.5,5.5))
    plt.scatter(y_true, y_hat, s=16, alpha=0.8)
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel(f"Observed {target_label}"); plt.ylabel(f"Predicted {target_label} ({name})")
    plt.title(f"Observed vs Predicted ({name}) — TEST")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=160)
    plt.close()


def main():
    ensure_packages()
    from PyGRF import PyGRF
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

    # 3) Filter rows (NaNs & invalid target)
    cols_needed = [TARGET_COL, LON_COL, LAT_COL] + FEATURE_COLS
    df = df.dropna(subset=cols_needed).reset_index(drop=True)
    df = df[df[TARGET_COL] >= 0]  # drop negative pm values if any
    if df.empty:
        raise ValueError("No valid rows after dropping NaNs/invalid values.")

    # 4) Build matrices
    y_all = df[TARGET_COL].astype(float)
    X_all = df[FEATURE_COLS].astype(float)
    coords_all = df[[LON_COL, LAT_COL]].astype(float).values
    n_all = len(df)

    # 5) Make spatial folds
    #folds = make_spatial_folds(coords_all, n_folds=N_FOLDS, n_clusters=N_CLUSTERS,min_test_fract=TEST_MIN_FRACT, random_state=RANDOM_STATE)
    folds = make_random_folds(len(df), n_folds=N_FOLDS, random_state=RANDOM_STATE)

    # Stores
    all_preds = []
    fold_metrics = []

    # RAM once
    avail_ram = get_available_ram_gb()

    for fold_id, (tr_idx, te_idx) in enumerate(folds, start=1):
        if te_idx.size == 0 or tr_idx.size == 0:
            print(f"[warn] Fold {fold_id} is empty; skipping.")
            continue

        X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
        y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]
        C_tr, C_te = coords_all[tr_idx], coords_all[te_idx]

        # ---- RF baseline on TRAIN ----
        rf = RandomForestRegressor(
            n_estimators=N_ESTIMATORS_RF,
            max_features=MAX_FEATURES_RF,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_RF
        )
        rf.fit(X_tr, y_tr)
        yhat_rf_tr = rf.predict(X_tr)
        yhat_rf_te = rf.predict(X_te)

        resid_rf_tr = y_tr.values - yhat_rf_tr
        resid_rf_te = y_te.values - yhat_rf_te

        w_rf_te = lps.weights.KNN.from_array(C_te, k=12); w_rf_te.transform = "r"
        moran_rf_te = Moran(resid_rf_te, w_rf_te)

        r2_rf_tr  = r2_score(y_tr, yhat_rf_tr);  mae_rf_tr = mean_absolute_error(y_tr, yhat_rf_tr)
        r2_rf_te  = r2_score(y_te, yhat_rf_te);  mae_rf_te = mean_absolute_error(y_te, yhat_rf_te)

        # ---- GRF (TRAIN only), RAM-aware per fold ----
        params = choose_grf_params(avail_ram, p_features=len(FEATURE_COLS), n_train=len(X_tr))
        builder = PyGRF.PyGRFBuilder(
            n_estimators=params["n_estimators"],
            max_features=params["max_features"],
            band_width=params["band_width"],
            train_weighted=params["train_weighted"],
            predict_weighted=params["predict_weighted"],
            bootstrap=BOOTSTRAP_DEFAULT,
            resampled=params["resampled"],
            random_state=RANDOM_STATE
        )
        builder.fit(X_tr, y_tr, C_tr)

        yhat_grf_tr, yhat_grf_tr_g, yhat_grf_tr_l = builder.predict(X_tr, C_tr, local_weight=params["local_weight"])
        yhat_grf_te, yhat_grf_te_g, yhat_grf_te_l = builder.predict(X_te, C_te, local_weight=params["local_weight"])

        resid_grf_tr = y_tr.values - yhat_grf_tr
        resid_grf_te = y_te.values - yhat_grf_te

        w_grf_te = lps.weights.KNN.from_array(C_te, k=12); w_grf_te.transform = "r"
        moran_grf_te = Moran(resid_grf_te, w_grf_te)

        r2_grf_tr  = r2_score(y_tr, yhat_grf_tr);  mae_grf_tr = mean_absolute_error(y_tr, yhat_grf_tr)
        r2_grf_te  = r2_score(y_te, yhat_grf_te);  mae_grf_te = mean_absolute_error(y_te, yhat_grf_te)

        # Save fold metrics
        fold_metrics.append({
            "fold": fold_id,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "rf_r2_train": r2_rf_tr,   "rf_mae_train": mae_rf_tr,
            "rf_r2_test":  r2_rf_te,   "rf_mae_test":  mae_rf_te,
            "rf_moranI_test": moran_rf_te.I, "rf_moranP_test": moran_rf_te.p_norm,
            "grf_r2_train": r2_grf_tr, "grf_mae_train": mae_grf_tr,
            "grf_r2_test":  r2_grf_te, "grf_mae_test":  mae_grf_te,
            "grf_moranI_test": moran_grf_te.I, "grf_moranP_test": moran_grf_te.p_norm,
            "grf_band_width": params["band_width"],
            "grf_n_estimators": params["n_estimators"],
            "grf_max_features": params["max_features"],
            "grf_resampled": params["resampled"]
        })

        # Predictions table for this fold
        fold_df = df.iloc[te_idx].copy()
        fold_df["fold"] = fold_id
        fold_df["split"] = "test"
        fold_df["pred_rf"]  = yhat_rf_te
        fold_df["resid_rf"] = resid_rf_te
        fold_df["pred_grf_combined"] = yhat_grf_te
        fold_df["pred_grf_global"]   = yhat_grf_te_g
        fold_df["pred_grf_local"]    = yhat_grf_te_l
        fold_df["resid_grf"]         = resid_grf_te
        all_preds.append(fold_df)

        # Plots for first fold (TEST)
        if fold_id == 1:
            # Histograms
            plt.figure(figsize=(6,4))
            plt.hist(resid_grf_te, bins=30, alpha=0.9); plt.axvline(0, color="k", lw=1)
            plt.title(f"GRF TEST residuals (fold {fold_id})\nR2={r2_grf_te:.3f}, MAE={mae_grf_te:.3f}, Moran's I={moran_grf_te.I:.3f} (p={moran_grf_te.p_norm:.4f})")
            plt.xlabel("Residual"); plt.ylabel("Count"); plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"residual_hist_grf_fold{fold_id}_test.png"), dpi=160); plt.close()

            plt.figure(figsize=(6,4))
            plt.hist(resid_rf_te, bins=30, alpha=0.9); plt.axvline(0, color="k", lw=1)
            plt.title(f"RF TEST residuals (fold {fold_id})\nR2={r2_rf_te:.3f}, MAE={mae_rf_te:.3f}, Moran's I={moran_rf_te.I:.3f} (p={moran_rf_te.p_norm:.4f})")
            plt.xlabel("Residual"); plt.ylabel("Count"); plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"residual_hist_rf_fold{fold_id}_test.png"), dpi=160); plt.close()

            # Obs vs Pred
            _obs_pred(y_te.values, yhat_grf_te, "GRF", f"obs_vs_pred_grf_fold{fold_id}_test.png",
                      OUTPUT_DIR, TARGET_COL)
            _obs_pred(y_te.values, yhat_rf_te, "RF", f"obs_vs_pred_rf_fold{fold_id}_test.png",
                      OUTPUT_DIR, TARGET_COL)

            # Residual map (GRF)
            plt.figure(figsize=(6,5))
            sc = plt.scatter(fold_df[LON_COL], fold_df[LAT_COL], c=fold_df["resid_grf"], s=18, cmap="RdBu", alpha=0.9)
            plt.colorbar(sc, label="Residual (GRF, TEST)")
            plt.title(f"Residuals map (GRF) — TEST fold {fold_id}")
            plt.xlabel("Longitude"); plt.ylabel("Latitude")
            plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, f"residuals_map_grf_fold{fold_id}_test.png"), dpi=160); plt.close()

            # RF feature importance & PDP (TRAIN)
            if hasattr(rf, "feature_importances_"):
                imp = np.array(rf.feature_importances_)
                order = np.argsort(imp)[::-1]
                plt.figure(figsize=(6,4))
                plt.bar(range(len(order)), imp[order])
                plt.xticks(range(len(order)), [FEATURE_COLS[i] for i in order], rotation=45, ha="right")
                plt.ylabel("Importance"); plt.title(f"RF feature importance (TRAIN — fold {fold_id})")
                plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, f"global_feature_importance_rf_fold{fold_id}.png"), dpi=160); plt.close()

                topk = min(3, len(FEATURE_COLS))
                feats = [FEATURE_COLS[i] for i in order[:topk]]
                try:
                    fig, ax = plt.subplots(1, topk, figsize=(5*topk, 4))
                    if topk == 1: ax = [ax]
                    PartialDependenceDisplay.from_estimator(rf, X_tr, feats, ax=ax)
                    fig.suptitle(f"RF partial dependence (TRAIN — fold {fold_id})")
                    plt.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, f"pdp_rf_top_features_fold{fold_id}.png"), dpi=160); plt.close(fig)
                except Exception:
                    pass

    # Save predictions across folds
    if all_preds:
        pd.concat(all_preds, ignore_index=True).to_csv(os.path.join(OUTPUT_DIR, "cv_predictions_residuals.csv"), index=False)
        print(f"[OK] Saved per-fold predictions/residuals: cv_predictions_residuals.csv")

    # Save fold metrics and summary
    dfm = pd.DataFrame(fold_metrics)
    dfm.to_csv(os.path.join(OUTPUT_DIR, "cv_fold_metrics.csv"), index=False)
    print(f"[OK] Saved fold metrics: cv_fold_metrics.csv")

    # Aggregate
    def mean_std(col):
        return dfm[col].mean(), dfm[col].std()
    with open(os.path.join(OUTPUT_DIR, "cv_summary.txt"), "w") as f:
        f.write(f"Spatial K-fold CV (K={N_FOLDS}, clusters={N_CLUSTERS})\n")
        f.write(f"Target: {TARGET_COL}\nFeatures: {FEATURE_COLS}\n")
        f.write(f"Samples: {n_all}\n\n")
        for model in ["rf", "grf"]:
            r2m, r2s = mean_std(f"{model}_r2_test")
            maem, maes = mean_std(f"{model}_mae_test")
            mim, mis = mean_std(f"{model}_moranI_test")
            f.write(f"{model.upper()} TEST: R2={r2m:.4f}±{r2s:.4f}, MAE={maem:.4f}±{maes:.4f}, Moran's I={mim:.4f}±{mis:.4f}\n")
        # GRF params range actually used
        if "grf_band_width" in dfm.columns:
            f.write("\nGRF params across folds (min–max):\n")
            f.write(f"  band_width:   {dfm['grf_band_width'].min()}–{dfm['grf_band_width'].max()}\n")
            f.write(f"  n_estimators: {dfm['grf_n_estimators'].min()}–{dfm['grf_n_estimators'].max()}\n")
            f.write(f"  max_features: {dfm['grf_max_features'].min()}–{dfm['grf_max_features'].max()}\n")
            f.write(f"  resampled:    {dfm['grf_resampled'].unique().tolist()}\n")

    print("[Done] Spatial K-fold CV — RF baseline + RAM-aware GRF complete.")

if __name__ == "__main__":
    main()
