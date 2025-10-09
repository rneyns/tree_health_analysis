
# ==============================================
# Quick visualisations script (save to PNG files)
# ==============================================

"""
Maak scatterplots met ongecontroleerde clustering (KMeans) en plot fenologische curves
op basis van de output-CSV's van het hoofdscript.

Gebruik:
- Pas de paden/instellingen onderaan aan en voer dit script uit (zelfde map is handig).
- Output gaat naar OUT_DIR (png-bestanden + een CSV met clusterlabels).
"""

# --------- Hard-coded paden & instellingen ---------
METRICS_CSV = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Statistical analysis/metrics_per_boom.csv"  # of bv. r"out/metrics_per_boom.csv"
TIMESERIES_CSV = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Statistical analysis/tijdreeksen_per_boom.csv" # of bv. r"out/tijdreeksen_per_boom.csv"
OUT_DIR = r"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Statistical analysis/out_figs"
N_CLUSTERS = 5
N_SAMPLES_PER_CLUSTER = 8
RANDOM_SEED = 42

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
except Exception as _e:
    StandardScaler = None
    KMeans = None
    silhouette_score = None
    PCA = None
    print("Waarschuwing: scikit-learn niet beschikbaar; clustering/ PCA is uitgeschakeld.")


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_and_prepare_metrics(path: str):
    df = pd.read_csv(path)
    # Verwachte numerieke metrics (pas aan als je meer wilt meenemen)
    num_cols = [
        'peak_value', 'peak_doy', 'SOS_doy', 'decline_slope_per_day'
    ]
    present = [c for c in num_cols if c in df.columns]
    sub = df[['tree_id'] + present].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    return df, sub, present


def cluster_metrics(sub: pd.DataFrame, feature_cols: list, n_clusters: int, seed: int):
    if KMeans is None or StandardScaler is None:
        sub['cluster'] = 0
        return sub, None, None
    X = sub[feature_cols].values
    n_clusters = max(2, min(n_clusters, len(sub)))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    labels = km.fit_predict(Xs)
    sub = sub.copy()
    sub['cluster'] = labels
    sil = None
    if silhouette_score is not None and len(np.unique(labels)) > 1 and len(sub) > n_clusters:
        with np.errstate(invalid='ignore'):
            sil = float(silhouette_score(Xs, labels))
    return sub, scaler, sil


def palette(n: int):
    # eenvoudige cyclische kleuren (laat matplotlib defaults bepalen)
    return list(range(n))


def scatter_pair_plots(sub: pd.DataFrame, feature_cols: list, out_dir: str, sil=None):
    clusters = sorted(sub['cluster'].unique())
    for x, y in combinations(feature_cols, 2):
        fig, ax = plt.subplots(figsize=(7, 5))
        for k in clusters:
            ss = sub[sub['cluster'] == k]
            ax.scatter(ss[x], ss[y], label=f"cluster {k}", alpha=0.8)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        title = f"{y} vs {x} (KMeans)"
        if sil is not None:
            title += f" — silhouette={sil:.2f}"
        ax.set_title(title)
        ax.legend(title="cluster")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"scatter_{y}_vs_{x}.png"), dpi=150)
        plt.close(fig)


def pca_scatter(sub: pd.DataFrame, feature_cols: list, out_dir: str, sil=None):
    if PCA is None or StandardScaler is None:
        return
    X = sub[feature_cols].values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    XY = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in sorted(sub['cluster'].unique()):
        ss = XY[sub['cluster'] == k]
        ax.scatter(ss[:, 0], ss[:, 1], label=f"cluster {k}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    var = pca.explained_variance_ratio_
    title = f"PCA scatter (PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%)"
    if sil is not None:
        title += f" — silhouette={sil:.2f}"
    ax.set_title(title)
    ax.legend(title="cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_scatter.png"), dpi=150)
    plt.close(fig)


def load_timeseries(path: str):
    ts = pd.read_csv(path)
    need = ['tree_id', 'doy']
    for c in need:
        if c not in ts.columns:
            raise ValueError(f"Kolom ontbreekt in timeseries: {c}")
    return ts


def phenology_overlay_means(ts: pd.DataFrame, clusters_df: pd.DataFrame, out_dir: str):
    # koppel clusters
    lab = clusters_df[['tree_id', 'cluster']]
    ts2 = ts.merge(lab, on='tree_id', how='inner')
    # Gebruik NDVI_fit als beschikbaar, anders NDVI
    ycol = 'NDVI_fit' if 'NDVI_fit' in ts2.columns and ts2['NDVI_fit'].notna().any() else 'NDVI'
    # Gemiddelde per cluster-per DOY
    mean_df = ts2.groupby(['cluster', 'doy'])[ycol].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, grp in mean_df.groupby('cluster'):
        ax.plot(grp['doy'], grp[ycol], label=f"cluster {k}")
    ax.set_xlabel('DOY')
    ax.set_ylabel(ycol)
    ax.set_title('Gemiddelde fenologie per cluster')
    ax.legend(title='cluster')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phenology_mean_by_cluster.png"), dpi=150)
    plt.close(fig)


def phenology_examples(ts: pd.DataFrame, clusters_df: pd.DataFrame, metrics_full: pd.DataFrame,
                       out_dir: str, n_per_cluster: int, seed: int):
    rng = np.random.default_rng(seed)
    lab = clusters_df[['tree_id', 'cluster']]
    ts2 = ts.merge(lab, on='tree_id', how='inner')
    # voor markers
    mcols = ['tree_id', 'peak_doy', 'SOS_doy']
    msub = metrics_full[mcols].dropna()

    for k in sorted(clusters_df['cluster'].unique()):
        ids = clusters_df[clusters_df['cluster'] == k]['tree_id'].unique()
        if len(ids) == 0:
            continue
        sel = rng.choice(ids, size=min(n_per_cluster, len(ids)), replace=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        for tid in sel:
            s = ts2[ts2['tree_id'] == tid].sort_values('doy')
            ycol = 'NDVI_fit' if 'NDVI_fit' in s.columns and s['NDVI_fit'].notna().any() else 'NDVI'
            ax.plot(s['doy'], s[ycol], alpha=0.9, label=f"tree {tid}")
            mm = msub[msub['tree_id'] == tid]
            if not mm.empty:
                pdoy = float(mm['peak_doy'].values[0])
                sdoy = float(mm['SOS_doy'].values[0])
                # Markers op curve (zoek dichtstbijzijnde DOY-index)
                if np.isfinite(pdoy):
                    ax.scatter(pdoy, np.interp(pdoy, s['doy'], s[ycol]), marker='^')
                if np.isfinite(sdoy):
                    ax.scatter(sdoy, np.interp(sdoy, s['doy'], s[ycol]), marker='x')
        ax.set_xlabel('DOY')
        ax.set_ylabel('NDVI (fit)')
        ax.set_title(f'Fenologie voorbeelden — cluster {k}')
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"phenology_examples_cluster_{k}.png"), dpi=150)
        plt.close(fig)


def run_viz():
    ensure_outdir(OUT_DIR)

    # 1) Metrics laden en clusteren
    metrics_full, sub, features = load_and_prepare_metrics(METRICS_CSV)
    clustered, scaler, sil = cluster_metrics(sub, features, N_CLUSTERS, RANDOM_SEED)

    # Bewaar clusterlabels
    narrow = clustered[['tree_id', 'cluster']]
    out_labels_csv = os.path.join(OUT_DIR, 'cluster_labels.csv')
    narrow.to_csv(out_labels_csv, index=False)

    # 2) Scatterplots en PCA
    scatter_pair_plots(clustered, features, OUT_DIR, sil=sil)
    pca_scatter(clustered, features, OUT_DIR, sil=sil)

    # 3) Fenologie plots
    ts = load_timeseries(TIMESERIES_CSV)
    phenology_overlay_means(ts, clustered, OUT_DIR)
    phenology_examples(ts, clustered, metrics_full, OUT_DIR, N_SAMPLES_PER_CLUSTER, RANDOM_SEED)

    print(f"Klaar. Figuren staan in: {OUT_DIR}, Clusterlabels: {out_labels_csv}")


# Voer only-out-viz uit als je dit script runt
if __name__ == '__main__':
    run_viz()