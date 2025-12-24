"""
This script generates exploratory scatterplots to visualise simple, unmodelled
relationships between urban environment variables and tree seasonal canopy
performance.

For each tree species (stored in a separate input CSV), the script produces:
1) NO₂ annual mean vs season-integrated NDVI (AUC), and
2) Impervious surface fraction vs season-integrated NDVI (AUC).

Plots are intended for qualitative assessment only and do not account for
covariates, spatial structure, or nonlinearity. They are used to complement
partial-correlation and GAM-based analyses by illustrating raw data structure,
variance, and potential species-specific response patterns.

Optional LOWESS smoothing is applied to highlight broad trends without imposing
a parametric model.
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import spearmanr
from pathlib import Path

# ----------------------------
# USER SETTINGS
# ----------------------------

DATA_FILES = {
    "Acer platanoides": '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/Acer platanoides/ndvi_metrics_with_impervious.csv',
    "Acer pseudoplatanus": '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/Acer pseudoplatanus/ndvi_metrics_with_impervious.csv',
    "Aesculus hippocastanum": '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/Aesculus hippocastanum /ndvi_metrics_with_impervious.csv',
    "Platanus × acerifolia": '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/Platanus x acerifolia/ndvi_metrics_with_impervious.csv',
    "Tilia × euchlora": '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/Tilia x euchlora/ndvi_metrics_with_impervious.csv',
}

AUC_COL = "auc_above_base_full"
NO2_COL = "poll_no2_anmean"
IMPERV_COL = "impervious_r50"   # change buffer if desired

ALPHA = 0.25
POINT_SIZE = 12
SMOOTH = True
SMOOTH_FRAC = 0.3

FIG_DPI = 300
OUTDIR = Path('/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Figures paper/scatterplot vars')
OUTDIR.mkdir(exist_ok=True)


# ----------------------------
# HELPERS
# ----------------------------

def safe_species_name(s: str) -> str:
    return s.replace(" ", "_").replace("×", "x").replace("/", "-")


def annotate_spearman(ax, x, y):
    """Compute and annotate Spearman rho on a plot."""
    if len(x) < 3:
        txt = "ρ = NA\nn < 3"
    else:
        rho, _ = spearmanr(x, y)
        txt = f"ρ = {rho:.2f}\nn = {len(x)}"

    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
    )


def scatter_simple(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    title: str,
    outfile: Path,
):
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    ax.scatter(df[xcol], df[ycol], s=POINT_SIZE, alpha=ALPHA)

    if SMOOTH and len(df) > 20:
        sm = lowess(df[ycol], df[xcol], frac=SMOOTH_FRAC, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1])

    annotate_spearman(ax, df[xcol], df[ycol])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.savefig(outfile, dpi=FIG_DPI)
    plt.close()


# ----------------------------
# MAIN LOOP
# ----------------------------

for species, path in DATA_FILES.items():
    print(f"Processing {species}  ({path})")

    df = pd.read_csv(path)

    # Remove invalid NO2 values
    if NO2_COL in df.columns:
        df = df[df[NO2_COL].isna() | (df[NO2_COL] >= 0)].copy()

    sp_tag = safe_species_name(species)

    # 1) NO2 vs AUC
    d1 = df.dropna(subset=[NO2_COL, AUC_COL])
    scatter_simple(
        df=d1,
        xcol=NO2_COL,
        ycol=AUC_COL,
        xlabel="NO₂ annual mean",
        ylabel="Season-integrated NDVI (AUC)",
        title=f"{species}: NO₂ vs seasonal NDVI (AUC)",
        outfile=OUTDIR / f"scatter_no2_vs_auc_{sp_tag}.png",
    )

    # 2) Impervious vs AUC
    d2 = df.dropna(subset=[IMPERV_COL, AUC_COL])
    scatter_simple(
        df=d2,
        xcol=IMPERV_COL,
        ycol=AUC_COL,
        xlabel=f"Impervious surface fraction ({IMPERV_COL})",
        ylabel="Season-integrated NDVI (AUC)",
        title=f"{species}: Impervious vs seasonal NDVI (AUC)",
        outfile=OUTDIR / f"scatter_impervious_vs_auc_{sp_tag}.png",
    )

    # 3) NO2 vs Impervious
    d3 = df.dropna(subset=[NO2_COL, IMPERV_COL])
    scatter_simple(
        df=d3,
        xcol=IMPERV_COL,
        ycol=NO2_COL,
        xlabel=f"Impervious surface fraction ({IMPERV_COL})",
        ylabel="NO₂ annual mean",
        title=f"{species}: NO₂ vs imperviousness",
        outfile=OUTDIR / f"scatter_no2_vs_impervious_{sp_tag}.png",
    )