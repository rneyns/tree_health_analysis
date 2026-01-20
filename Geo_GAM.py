#!/usr/bin/env python3
"""
Multi-species GAM workflow focused on:

    "Is there remaining spatial structure in the residuals after accounting for our regressor variables?"

For each species CSV and each health metric:
1) Fit a NON-spatial GAM (regressors + controls)
2) Fit a SPATIAL GAM (same + spatial smooth of coordinates)
3) Use K-fold CV to get out-of-fold (OOF) predictions for each model
4) Compute residuals (OOF): resid = y - yhat
5) Compute Moran's I on residuals (KNN weights) + permutation p-value
6) Save results + figures comparing Moran's I:
   - per-species paired "dumbbell" plot per health metric: I_nonspatial vs I_spatial
   - per-species bar plot of ΔI = I_nonspatial - I_spatial
   - combined overview across species: mean ΔI per species

IMPORTANT:
- If your CSVs do NOT contain x/y, this script will JOIN coordinates from a shapefile using an ID key.
- You MUST set SHAPEFILE_PATH, CSV_ID_COL, SHP_ID_COL, and (ideally) TARGET_CRS below.

Dependencies:
  pip install numpy pandas matplotlib scikit-learn pygam geopandas shapely pyproj fiona
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd

from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from pygam import LinearGAM, s

try:
    from pygam import te  # optional true 2D smooth
    HAS_TE = True
except Exception:
    HAS_TE = False


# ----------------------------
# USER SETTINGS
# ----------------------------

SPECIES_CSVS = {
    "Acer_platanoides": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer platanoides/ndvi_metrics_with_impervious.csv",
    "Acer_pseudoplatanus": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer pseudoplatanus/ndvi_metrics_with_impervious.csv",
    "Aesculus_hippocastanum": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/aesculus hippocastanum/ndvi_metrics_with_impervious.csv",
    "Platanus_x_acerifolia": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/platanus x acerifolia/ndvi_metrics_with_impervious.csv",
    "Tilia_x_euchlora": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/tilia x euchlora/ndvi_metrics_with_impervious.csv",
}

# Pretty names for plots/tables
SPECIES_LABELS = {
    "Acer_platanoides": "Acer platanoides",
    "Acer_pseudoplatanus": "Acer pseudoplatanus",
    "Aesculus_hippocastanum": "Aesculus hippocastanum",
    "Platanus_x_acerifolia": "Platanus × acerifolia",
    "Tilia_x_euchlora": "Tilia × euchlora",
}

HEALTH_LABELS = {
    "ndvi_peak": "Peak NDVI",
    "sos_doy": "Start of season (DOY)",
    "los_days": "Length of season (days)",
    "amplitude": "NDVI amplitude",
    "auc_above_base_full": "Seasonal NDVI integral",
}

PREDICTOR_LABELS = {
    "imperv_10m": "Impervious surface (10 m)",
    "poll_bc_anmean": "Black carbon (annual mean)",
    "lst_temp_r50_y": "Land surface temperature (50 m)",
    "insolation9": "Solar radiation",
}

OUT_ROOT = Path("/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/_GAM_MULTI_SPECIES_SPATIAL_RESIDUALS")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Model variables (raw codes)
CONTROL_VARS = ["height"]
HEALTH_VARS = ["ndvi_peak", "sos_doy", "los_days", "amplitude", "auc_above_base_full"]

PREDICTORS = [
    "imperv_10m",
    "poll_bc_anmean",
    "lst_temp_r50_y",
    "insolation9",
]

EXTRA_CONTROLS = []  # e.g. ["imperv_50m"]

# Derived pretty order list (IMPORTANT for plots after relabeling)
HEALTH_VARS_PRETTY = [HEALTH_LABELS.get(h, h) for h in HEALTH_VARS]

# ---------- COORDINATES ----------
# If CSV already contains coords, set these to match the CSV columns.
# If CSV does NOT contain coords, set these to ['x','y'] and the script will attach them from shapefile.
COORD_COLS = ["x", "y"]

# ---------- SHAPEFILE JOIN SETTINGS ----------
SHAPEFILE_PATH = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/flai layers/crown_shapes_final_CRS.shp"
CSV_ID_COL = "tree_id"
SHP_ID_COL = "crown_id"
USE_CENTROID_IF_NOT_POINTS = True

# Reproject shapefile to a metric CRS for KNN (recommended).
# If your shapefile is already projected in meters, set to None.
TARGET_CRS = None  # e.g. "EPSG:31370" or "EPSG:32631"

# GAM settings
N_SPLINES_1D = 10
N_SPLINES_2D = 20  # only used if te() exists
LAM_GRID = np.logspace(-3, 3, 13)
DO_GRIDSEARCH = True

# CV settings
N_FOLDS = 5
RANDOM_STATE = 42

# Moran settings
K_NEIGHBORS = 8
N_PERMUTATIONS = 999
SEED_MORAN = 42

# Data cleaning / filtering
MIN_N = 80
STANDARDIZE = True
STANDARDIZE_COORDS = True
EPS = 1e-12

BIOLOGICAL_FILTERS = {
    "los_days": lambda x: (x >= 60) & (x <= 365),
    "sos_doy": lambda x: (x >= 1) & (x <= 250),
    "ndvi_peak": lambda x: (x >= -0.1) & (x <= 1.0),
    "amplitude": lambda x: (x >= -0.1) & (x <= 1.0),
    "auc_above_base_full": lambda x: x > -1e9,
    "height": lambda x: x > 1,
    "poll_bc_anmean": lambda x: x > 0,
}

# Plot settings
FIG_DPI = 240
POINT_ALPHA = 0.9


# ----------------------------
# Helpers
# ----------------------------

def apply_labels(series: pd.Series, label_map: dict) -> pd.Series:
    """Replace values in a Series using label_map; keep original if not found."""
    return series.map(lambda x: label_map.get(x, x))

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def zscore(s: pd.Series, eps: float = 1e-12) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd < eps:
        return (s - mu) * 0.0
    return (s - mu) / sd

def make_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def apply_filters(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in BIOLOGICAL_FILTERS and c in out.columns:
            out = out[BIOLOGICAL_FILTERS[c](out[c])]
    return out

def _normalize_id_numeric(s: pd.Series) -> pd.Series:
    """
    Try to normalize IDs to numeric Int64 (nullable) safely.
    - Handles '123', '123.0', 123, 123.0
    - Non-numeric becomes <NA>
    """
    # to numeric float
    x = pd.to_numeric(s, errors="coerce")
    # drop fractional parts only if very close to integer
    # (this avoids turning a true decimal ID into an int)
    frac = np.abs(x - np.round(x))
    x = x.where((~x.isna()) & (frac < 1e-9), x)  # keep original for non-integerish
    x_int = np.round(x).astype("Int64")
    return x_int

def _normalize_id_str(s: pd.Series) -> pd.Series:
    """
    String normalization fallback:
    - strips whitespace
    - removes trailing '.0' repeatedly (common from float->str)
    - lowercases
    """
    out = s.astype(str).str.strip()
    # remove common float artifact '.0'
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.replace(r"\s+", "", regex=True)
    out = out.str.lower()
    # treat 'nan' literal as missing
    out = out.replace("nan", pd.NA)
    return out

def attach_xy_from_shapefile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach COORD_COLS (x,y) to df.
    - If df already has both coord cols, returns df unchanged.
    - Else merges coords from SHAPEFILE_PATH using CSV_ID_COL <-> SHP_ID_COL.

    Robust to common ID problems:
    - 123 vs 123.0
    - int vs float vs string
    - whitespace
    """
    if all(c in df.columns for c in COORD_COLS):
        return df

    if SHAPEFILE_PATH is None or str(SHAPEFILE_PATH).strip() == "":
        raise ValueError(f"CSV is missing coords {COORD_COLS} and SHAPEFILE_PATH is not set.")

    if CSV_ID_COL not in df.columns:
        raise ValueError(f"CSV is missing join key column '{CSV_ID_COL}' required to attach coords.")

    gdf = gpd.read_file(SHAPEFILE_PATH)

    if SHP_ID_COL not in gdf.columns:
        raise ValueError(f"Shapefile is missing join key column '{SHP_ID_COL}'.")

    if gdf.geometry is None:
        raise ValueError("Shapefile has no geometry column.")

    if TARGET_CRS is not None:
        gdf = gdf.to_crs(TARGET_CRS)

    geom = gdf.geometry
    if USE_CENTROID_IF_NOT_POINTS:
        geom = geom.centroid

    xname, yname = COORD_COLS[0], COORD_COLS[1]
    gdf_xy = gdf[[SHP_ID_COL]].copy()
    gdf_xy[xname] = geom.x
    gdf_xy[yname] = geom.y

    # ---- Robust join: try numeric join first, fallback to string join ----
    df2 = df.copy()

    # Numeric attempt
    df2["_join_num"] = _normalize_id_numeric(df2[CSV_ID_COL])
    gdf_xy2 = gdf_xy.copy()
    gdf_xy2["_join_num"] = _normalize_id_numeric(gdf_xy2[SHP_ID_COL])

    merged_num = df2.merge(
        gdf_xy2[["_join_num", xname, yname]],
        how="left",
        on="_join_num",
        suffixes=("", "_shp")
    )

    # If numeric join matched at least some rows, use it
    n_matched = merged_num[[xname, yname]].notna().all(axis=1).sum()

    if n_matched == 0:
        # String fallback
        df2["_join_str"] = _normalize_id_str(df2[CSV_ID_COL])
        gdf_xy2["_join_str"] = _normalize_id_str(gdf_xy2[SHP_ID_COL])

        merged_str = df2.merge(
            gdf_xy2[["_join_str", xname, yname]],
            how="left",
            on="_join_str",
            suffixes=("", "_shp")
        )
        out = merged_str.drop(columns=["_join_num", "_join_str"])
    else:
        out = merged_num.drop(columns=["_join_num"])

    return out

def build_terms_1d(n_cols: int) -> object:
    terms = None
    for j in range(n_cols):
        t = s(j, n_splines=N_SPLINES_1D)
        terms = t if terms is None else (terms + t)
    return terms

def knn_weights(xy: np.ndarray, k: int) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, idx = nbrs.kneighbors(xy)
    idx = idx[:, 1:]  # drop self
    n = xy.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        W[i, idx[i]] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    return W

def morans_I(resid: np.ndarray, W: np.ndarray) -> float:
    z = resid - np.nanmean(resid)
    n = z.size
    den = z @ z
    if not np.isfinite(den) or den <= 0:
        return np.nan
    num = z @ (W @ z)
    return (n / W.sum()) * (num / den)

def morans_I_perm(resid: np.ndarray, W: np.ndarray, n_perm: int = 999, seed: int = 42):
    rng = np.random.default_rng(seed)
    I_obs = morans_I(resid, W)
    if not np.isfinite(I_obs):
        return I_obs, np.nan

    sims = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        sims[b] = morans_I(rng.permutation(resid), W)

    p = (np.sum(np.abs(sims) >= np.abs(I_obs)) + 1) / (n_perm + 1)
    return I_obs, p

def cv_predict_gam(X: np.ndarray, y: np.ndarray, xy: np.ndarray | None, add_spatial: bool) -> np.ndarray:
    """OOF predictions for nonspatial GAM(X) or spatial GAM([X,xy]) + spatial smooth."""
    yhat = np.full_like(y, np.nan, dtype=float)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    terms_x = build_terms_1d(X.shape[1])

    for tr, te_idx in kf.split(X):
        if add_spatial:
            if xy is None:
                raise ValueError("xy required when add_spatial=True")

            Xtr = np.column_stack([X[tr], xy[tr]])
            Xte = np.column_stack([X[te_idx], xy[te_idx]])

            ix = X.shape[1]
            iy = X.shape[1] + 1

            if HAS_TE:
                terms = terms_x + te(ix, iy, n_splines=N_SPLINES_2D)
            else:
                terms = terms_x + s(ix, n_splines=N_SPLINES_1D) + s(iy, n_splines=N_SPLINES_1D)

            gam = LinearGAM(terms)
        else:
            Xtr = X[tr]
            Xte = X[te_idx]
            gam = LinearGAM(terms_x)

        if DO_GRIDSEARCH:
            gam.gridsearch(Xtr, y[tr], lam=LAM_GRID, progress=False)
        else:
            gam.fit(Xtr, y[tr])

        yhat[te_idx] = gam.predict(Xte)

    return yhat

def compute_moran_for_health(df_sub: pd.DataFrame, health: str, predictors: list[str], controls: list[str]) -> pd.DataFrame:
    cols_x = predictors + controls + EXTRA_CONTROLS
    needed = [health] + cols_x + COORD_COLS

    d = df_sub[needed].dropna().copy()
    if len(d) < MIN_N:
        return pd.DataFrame()

    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    if len(d) < MIN_N:
        return pd.DataFrame()

    if STANDARDIZE:
        d[health] = zscore(d[health], eps=EPS)
        for c in cols_x:
            d[c] = zscore(d[c], eps=EPS)

    if STANDARDIZE_COORDS:
        for c in COORD_COLS:
            d[c] = zscore(d[c], eps=EPS)

    y = d[health].values.astype(float)
    X = d[cols_x].values.astype(float)
    xy = d[COORD_COLS].values.astype(float)

    if len(d) <= (K_NEIGHBORS + 2):
        return pd.DataFrame()

    W = knn_weights(xy, k=K_NEIGHBORS)

    yhat_non = cv_predict_gam(X, y, xy=None, add_spatial=False)
    resid_non = y - yhat_non

    yhat_sp = cv_predict_gam(X, y, xy=xy, add_spatial=True)
    resid_sp = y - yhat_sp

    I_non, p_non = morans_I_perm(resid_non, W, n_perm=N_PERMUTATIONS, seed=SEED_MORAN)
    I_sp, p_sp = morans_I_perm(resid_sp, W, n_perm=N_PERMUTATIONS, seed=SEED_MORAN)

    out = pd.DataFrame([
        {"health_metric": health, "model": "nonspatial", "morans_I": I_non, "p_perm": p_non, "n": len(d),
         "k_neighbors": K_NEIGHBORS, "permutations": N_PERMUTATIONS},
        {"health_metric": health, "model": "spatial_smooth", "morans_I": I_sp, "p_perm": p_sp, "n": len(d),
         "k_neighbors": K_NEIGHBORS, "permutations": N_PERMUTATIONS},
    ])
    out["delta_I_non_minus_spatial"] = I_non - I_sp
    return out


# ----------------------------
# Plotting
# ----------------------------

def dumbbell_plot_moran(df_m: pd.DataFrame, title: str, outpath: Path):
    """Paired dot plot: Moran's I nonspatial vs spatial_smooth per health metric."""
    if df_m.empty:
        return

    piv = df_m.pivot_table(index="health_metric", columns="model", values="morans_I", aggfunc="mean")
    piv = piv.reindex(HEALTH_VARS_PRETTY)  # IMPORTANT: pretty order after relabel
    piv = piv.dropna(how="all")
    if piv.empty:
        return

    healths = piv.index.tolist()
    y = np.arange(len(healths))

    non = piv["nonspatial"].values if "nonspatial" in piv.columns else np.full(len(healths), np.nan)
    sp = piv["spatial_smooth"].values if "spatial_smooth" in piv.columns else np.full(len(healths), np.nan)

    plt.figure(figsize=(7.2, 0.7 + 0.6 * len(healths)))
    ax = plt.gca()

    for i in range(len(healths)):
        if np.isfinite(non[i]) and np.isfinite(sp[i]):
            ax.plot([non[i], sp[i]], [y[i], y[i]], linewidth=2, alpha=0.8)

    ax.scatter(non, y, s=90, alpha=POINT_ALPHA, label="Non-spatial GAM")
    ax.scatter(sp, y, s=90, alpha=POINT_ALPHA, label="Spatial-smooth GAM")

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(healths)
    ax.set_xlabel("Moran's I of OOF residuals")
    ax.set_title(title)
    #ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()

def delta_barplot(df_m: pd.DataFrame, title: str, outpath: Path):
    """Bar plot of ΔI = I_nonspatial - I_spatial per health metric."""
    if df_m.empty:
        return

    piv = df_m.pivot_table(index="health_metric", columns="model", values="morans_I", aggfunc="mean")
    piv = piv.reindex(HEALTH_VARS_PRETTY)  # IMPORTANT: pretty order after relabel
    piv = piv.dropna(how="all")
    if piv.empty or "nonspatial" not in piv.columns or "spatial_smooth" not in piv.columns:
        return

    delta = (piv["nonspatial"] - piv["spatial_smooth"]).copy()

    plt.figure(figsize=(6.8, 3.6))
    plt.bar(delta.index.tolist(), delta.values)
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.ylabel("Δ Moran's I (nonspatial − spatial)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()

def combined_overview_species(df_all: pd.DataFrame, outpath: Path):
    """Combined overview: mean ΔI per species (averaged across health metrics)."""
    if df_all.empty:
        return

    piv = df_all.pivot_table(index=["species", "health_metric"], columns="model", values="morans_I", aggfunc="mean")
    if "nonspatial" not in piv.columns or "spatial_smooth" not in piv.columns:
        return

    piv["delta_I"] = piv["nonspatial"] - piv["spatial_smooth"]
    piv = piv.reset_index()

    agg = (piv.groupby("species", as_index=False)
           .agg(delta_I_mean=("delta_I", "mean"),
                delta_I_median=("delta_I", "median"),
                delta_I_std=("delta_I", "std"),
                n_health=("delta_I", "count"))
           .sort_values("delta_I_mean", ascending=False))

    plt.figure(figsize=(7.6, 0.7 + 0.55 * len(agg)))
    ax = plt.gca()
    ax.barh(agg["species"], agg["delta_I_mean"])
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Mean Δ Moran's I (nonspatial − spatial)")
    ax.set_title("Residual spatial structure removed by spatial smooth (mean across health metrics)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()


# ----------------------------
# MAIN
# ----------------------------

def main():
    all_rows = []

    for species, csv_path in SPECIES_CSVS.items():
        print(f"\n=== {species} ===")
        sp_out = OUT_ROOT / species
        ensure_dir(sp_out)

        df = pd.read_csv(csv_path)

        required = HEALTH_VARS + PREDICTORS + CONTROL_VARS + EXTRA_CONTROLS
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[skip] Missing columns in {species}: {missing}")
            continue

        # Attach coordinates
        try:
            df = attach_xy_from_shapefile(df)
        except Exception as e:
            print(f"[skip] {species}: could not attach coords ({e})")
            continue

        if not all(c in df.columns for c in COORD_COLS):
            print(f"[skip] {species}: coords columns still missing after attach: {COORD_COLS}")
            continue

        if df[COORD_COLS].isna().all().all():
            print(f"[skip] {species}: coord join produced all-NaN coords (ID mismatch?)")
            continue

        df = make_numeric(df, required + COORD_COLS)
        df = apply_filters(df, required + COORD_COLS)

        sp_pretty = SPECIES_LABELS.get(species, species)

        species_results = []
        for health in HEALTH_VARS:
            df_m = compute_moran_for_health(df, health, PREDICTORS, CONTROL_VARS)
            if df_m.empty:
                print(f"[warn] {species}: no Moran result for {health} (n too small?)")
                continue

            df_m["species"] = sp_pretty
            df_m["health_metric"] = apply_labels(df_m["health_metric"], HEALTH_LABELS)

            species_results.append(df_m)
            all_rows.append(df_m)

        if not species_results:
            print(f"[skip] {species}: no health metric produced results")
            continue

        sp_res = pd.concat(species_results, ignore_index=True)

        sp_res.to_csv(sp_out / "moran_residuals_nonspatial_vs_spatial_by_health.csv", index=False)

        dumbbell_plot_moran(
            sp_res,
            title=f"{sp_pretty}: Moran's I of OOF residuals (nonspatial vs spatial smooth)",
            outpath=sp_out / "moran_dumbbell_by_health.png"
        )
        delta_barplot(
            sp_res,
            title=f"{sp_pretty}: Δ Moran's I (nonspatial − spatial) by health metric",
            outpath=sp_out / "moran_delta_barplot_by_health.png"
        )

        pivI = sp_res.pivot_table(index="health_metric", columns="model", values="morans_I", aggfunc="mean")
        pivP = sp_res.pivot_table(index="health_metric", columns="model", values="p_perm", aggfunc="mean")
        out_piv = pivI.copy()
        out_piv.columns = [f"I_{c}" for c in out_piv.columns]
        out_piv = out_piv.join(pivP.rename(columns=lambda c: f"p_{c}"))
        if "I_nonspatial" in out_piv.columns and "I_spatial_smooth" in out_piv.columns:
            out_piv["delta_I_non_minus_spatial"] = out_piv["I_nonspatial"] - out_piv["I_spatial_smooth"]
        out_piv = out_piv.reindex(HEALTH_VARS_PRETTY)
        out_piv.to_csv(sp_out / "moran_residuals_pivot.csv")

        print(f"[ok] saved: {sp_out}")

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_df.to_csv(OUT_ROOT / "ALL_species_moran_residuals_long.csv", index=False)
        combined_overview_species(all_df, OUT_ROOT / "ALL_species_mean_delta_moran_overview.png")
        print(f"\n[ok] multi-species outputs in: {OUT_ROOT}")
    else:
        print("\n[warn] No species produced outputs (check MIN_N, join IDs, shapefile path, or filters).")


if __name__ == "__main__":
    main()
