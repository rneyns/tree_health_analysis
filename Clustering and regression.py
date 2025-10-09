#!/usr/bin/env python3
"""
EDA clustering for phenology metrics (PM2.5 excluded from clustering),
and write cluster labels back to the tree layer.

Input
-----
- metrics_with_pm.csv  (must contain: sos_doy, peak_doy, ndvi_peak, slope_sos_peak, pm25; 'tree_id' optional)
- tree layer (SHP/GPKG/GeoJSON). If it lacks 'tree_id', the script adds it by position (0..N-1).

Outputs (in OUTPUT_DIR)
-----------------------
- merged_for_clustering.csv
- kmeans_elbow.png, kmeans_silhouette.png
- pca_scatter_kmeans.png, pca_scatter_hierarchical.png
- pm25_by_cluster_boxplot.png, cluster_summary.csv
- clusters_map.png  (optional quick plot)
- OUTPUT_TREE_LAYER (tree layer with 'tree_id' + cluster fields)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

import geopandas as gpd

# ------------------------- CONFIG -------------------------
METRICS_WITH_PM_CSV = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/ndvi_metrics.csv'     # CSV with phenology metrics (+ optional tree_id) and pm25
OUTPUT_DIR          = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/output_clustering'

# Cluster ONLY on phenology/health metrics (exclude pm25 here)
CLUSTER_FEATURES = [
    "sos_doy",
    "peak_doy",
    "ndvi_peak",
    "slope_sos_peak",
]

# Columns to include in the per-cluster summary (you can add more)
SUMMARY_COLS = CLUSTER_FEATURES + ["pm25"]

# REQUIRED: input tree layer (SHP/GPKG/GeoJSON)
TREE_LAYER_PATH = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Environmental variables/tree layers/platanus_x_acerifolia.shp'      # or .shp / .geojson
TREE_LAYER_NAME = None                    # set if GPKG; else None for SHP/GeoJSON

# Where to save the tree layer WITH clusters
OUTPUT_TREE_LAYER       = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/trees_with_clusters.gpkg"  # extension determines driver
OUTPUT_TREE_LAYER_NAME  = "trees_clustered"                    # required if GPKG; ignored for SHP/GeoJSON

# k-means scan range and chosen k
K_RANGE   = range(2, 9)   # try k=2..8 for elbow/silhouette
KMEANS_K  = 4             # final k to use for labeling/plots

# PCA settings (just for visualization)
PCA_COMPONENTS = 2
RANDOM_STATE   = 42
# ----------------------------------------------------------


def load_data():
    df = pd.read_csv(METRICS_WITH_PM_CSV)
    # Ensure required feature columns exist
    needed = CLUSTER_FEATURES + ["pm25"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def scale_features(df):
    X = df[CLUSTER_FEATURES].astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def kmeans_scan(Xs, out_dir):
    inertias, silhouettes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xs, labels))

    # Elbow
    plt.figure(figsize=(6,4))
    plt.plot(list(K_RANGE), inertias, marker="o")
    plt.xlabel("k"); plt.ylabel("Inertia")
    plt.title("K-means Elbow"); plt.grid(alpha=0.3); plt.tight_layout()
    elbow_path = os.path.join(out_dir, "kmeans_elbow.png")
    plt.savefig(elbow_path, dpi=160); plt.close()

    # Silhouette
    plt.figure(figsize=(6,4))
    plt.plot(list(K_RANGE), silhouettes, marker="o")
    plt.xlabel("k"); plt.ylabel("Average silhouette")
    plt.title("K-means Silhouette"); plt.grid(alpha=0.3); plt.tight_layout()
    sil_path = os.path.join(out_dir, "kmeans_silhouette.png")
    plt.savefig(sil_path, dpi=160); plt.close()

    print(f"[OK] Saved elbow: {elbow_path}")
    print(f"[OK] Saved silhouette: {sil_path}")


def run_clustering_and_pca(Xs, df, out_dir):
    # K-means final (labels based ONLY on CLUSTER_FEATURES)
    kmeans = KMeans(n_clusters=KMEANS_K, n_init=50, random_state=RANDOM_STATE)
    df["cluster_kmeans"] = kmeans.fit_predict(Xs)

    # Hierarchical (Ward)
    hier = AgglomerativeClustering(n_clusters=KMEANS_K, linkage="ward")
    df["cluster_hier"] = hier.fit_predict(Xs)

    # PCA (viz; also ONLY on CLUSTER_FEATURES)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xs)
    df["PC1"] = Xp[:,0]; df["PC2"] = Xp[:,1]

    # PCA colored by kmeans
    plt.figure(figsize=(6,5))
    for c in sorted(df["cluster_kmeans"].unique()):
        sub = df[df["cluster_kmeans"] == c]
        plt.scatter(sub["PC1"], sub["PC2"], s=22, alpha=0.9, label=f"Cluster {c}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title(f"PCA (k-means, k={KMEANS_K})")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    pca_kmeans_path = os.path.join(out_dir, "pca_scatter_kmeans.png")
    plt.savefig(pca_kmeans_path, dpi=160); plt.close()

    # PCA colored by hierarchical
    plt.figure(figsize=(6,5))
    for c in sorted(df["cluster_hier"].unique()):
        sub = df[df["cluster_hier"] == c]
        plt.scatter(sub["PC1"], sub["PC2"], s=22, alpha=0.9, label=f"Cluster {c}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title(f"PCA (Hierarchical, k={KMEANS_K})")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    pca_hier_path = os.path.join(out_dir, "pca_scatter_hierarchical.png")
    plt.savefig(pca_hier_path, dpi=160); plt.close()

    print(f"[OK] Saved PCA plots:\n- {pca_kmeans_path}\n- {pca_hier_path}")
    return df


def pm25_boxplot(df, out_dir):
    """Boxplot of PM2.5 by k-means cluster (PM2.5 was NOT used for clustering)."""
    if "pm25" not in df.columns or "cluster_kmeans" not in df.columns:
        return
    plt.figure(figsize=(6.5,5))
    data = [df[df["cluster_kmeans"] == c]["pm25"].dropna().values
            for c in sorted(df["cluster_kmeans"].unique())]
    plt.boxplot(data, labels=[f"C{c}" for c in sorted(df["cluster_kmeans"].unique())], showmeans=True)
    plt.xlabel("Cluster (k-means)")
    plt.ylabel("PM2.5")
    plt.title("PM2.5 by cluster (excluded from clustering)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "pm25_by_cluster_boxplot.png")
    plt.savefig(out_path, dpi=160); plt.close()
    print(f"[OK] Saved PM2.5 boxplot: {out_path}")


def cluster_summary(df, out_dir):
    """Save per-cluster mean/std/count for selected columns (incl. pm25)."""
    group = df.groupby("cluster_kmeans")
    means = group[SUMMARY_COLS].mean().add_suffix("_mean")
    stds  = group[SUMMARY_COLS].std().add_suffix("_std")
    cnt   = group.size().to_frame("n")
    summary = pd.concat([cnt, means, stds], axis=1).reset_index()
    out_csv = os.path.join(out_dir, "cluster_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"[OK] Saved per-cluster summary: {out_csv}")


def write_clusters_to_layer(df_with_labels: pd.DataFrame):
    """
    Join cluster labels to the tree layer.
    - If the tree layer has 'tree_id', join by that field.
    - Else, assume the feature order matches the CSV row order, create a new 'tree_id' (0..N-1),
      align by position, and write the output.
    """
    # Load tree layer
    trees = gpd.read_file(TREE_LAYER_PATH, layer=TREE_LAYER_NAME) if TREE_LAYER_NAME else gpd.read_file(TREE_LAYER_PATH)

    # set these once (top of script / CONFIG)
    LAYER_ID_FIELD = "crown_id"  # the id column in the tree layer
    CSV_ID_FIELD = "tree_id"  # the id column in df_with_labels

    # --- robust join that handles different id names + fallbacks ---
    trees_has_id = LAYER_ID_FIELD in trees.columns
    csv_has_id = CSV_ID_FIELD in df_with_labels.columns

    if trees_has_id and csv_has_id:
        # rename both to a common key and merge
        trees_key = trees.rename(columns={LAYER_ID_FIELD: "__id"})
        labels_key = df_with_labels.rename(columns={CSV_ID_FIELD: "__id"})
        keep = ["__id", "cluster_kmeans", "cluster_hier"]
        gdf_out = trees_key.merge(labels_key[keep].drop_duplicates("__id"), on="__id", how="left")
        # restore original layer id name
        gdf_out = gdf_out.rename(columns={"__id": LAYER_ID_FIELD})

    # Decide driver by extension
    ext = os.path.splitext(OUTPUT_TREE_LAYER)[1].lower()
    if ext == ".gpkg":
        if os.path.exists(OUTPUT_TREE_LAYER):
            os.remove(OUTPUT_TREE_LAYER)  # clean write
        gdf_out.to_file(OUTPUT_TREE_LAYER, layer=OUTPUT_TREE_LAYER_NAME, driver="GPKG")
    else:
        if os.path.exists(OUTPUT_TREE_LAYER):
            os.remove(OUTPUT_TREE_LAYER)
        gdf_out.to_file(OUTPUT_TREE_LAYER)

    print(f"[OK] Wrote tree layer with clusters → {OUTPUT_TREE_LAYER}" +
          (f"::{OUTPUT_TREE_LAYER_NAME}" if ext == ".gpkg" else ""))


def draw_map(df, out_dir):
    try:
        trees = gpd.read_file(TREE_LAYER_PATH, layer=TREE_LAYER_NAME) if TREE_LAYER_NAME else gpd.read_file(TREE_LAYER_PATH)
    except Exception as e:
        print(f"[skip] Could not read tree layer: {e}")
        return

    # Align by available key
    if "tree_id" in trees.columns and "tree_id" in df.columns:
        gdf = trees.merge(df[["tree_id", "cluster_kmeans"]], on="tree_id", how="left")
    else:
        # by position (best-effort preview)
        if len(trees) != len(df):
            print("[skip] Map join by position failed: different lengths.")
            return
        gdf = trees.reset_index(drop=True).copy()
        gdf["cluster_kmeans"] = df["cluster_kmeans"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7,6))
    if gdf.geometry.is_empty.all():
        print("[skip] tree layer has no geometries.")
        return

    clusters = sorted([c for c in gdf["cluster_kmeans"].dropna().unique()])
    for c in clusters:
        gdf[gdf["cluster_kmeans"] == c].plot(ax=ax, markersize=8, label=f"Cluster {int(c)}", alpha=0.9)

    ax.set_title(f"Trees colored by k-means cluster (k={KMEANS_K})")
    ax.set_axis_off()
    ax.legend()
    map_path = os.path.join(out_dir, "clusters_map.png")
    plt.tight_layout(); plt.savefig(map_path, dpi=160); plt.close()
    print(f"[OK] Saved map: {map_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load CSV with metrics (+/- tree_id) and pm25
    df = load_data()

    # 2) Filter rows that have finite values for the clustering features
    mask = np.ones(len(df), dtype=bool)
    for c in CLUSTER_FEATURES:
        mask &= np.isfinite(df[c].astype(float))
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after filtering for finite phenology metrics.")

    # 3) Scale ONLY the clustering features
    Xs, _ = scale_features(df)

    # 4) Scan k (EDA)
    kmeans_scan(Xs, OUTPUT_DIR)

    # 5) Final clustering + PCA (based ONLY on phenology metrics)
    df_out = run_clustering_and_pca(Xs, df.copy(), OUTPUT_DIR)

    # 6) Save merged table with labels
    out_csv = os.path.join(OUTPUT_DIR, "merged_for_clustering.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[OK] Saved merged table with clusters: {out_csv}")

    # 7) PM2.5 by cluster (diagnostic) + per-cluster summary
    pm25_boxplot(df_out, OUTPUT_DIR)
    cluster_summary(df_out, OUTPUT_DIR)

    # 8) Optional quick map (joins by tree_id when present; else by position)
    draw_map(df_out, OUTPUT_DIR)

    # 9) Write cluster labels back to the tree layer
    write_clusters_to_layer(df_out)

    print("[Done] EDA clustering + layer export complete.")

if __name__ == "__main__":
    main()
