import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations, product
from statsmodels.nonparametric.smoothers_lowess import lowess
import pingouin as pg
from statsmodels.stats.multitest import multipletests  # for optional FDR correction

# ----------------------------
# USER SETTINGS
# ----------------------------

DATA_PATH = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/ndvi_metrics_with_impervious.csv"

health_vars = [
    "ndvi_peak",
    "ndvi_base",
    "greenup_doy",
    "sos_doy",
    "peak_doy",
    "sen_onset_doy",
    "eos_doy",
    "dormancy_doy",
    "los_days",
    "amplitude",
    "slope_sos_peak",
    "senescence_rate",
    "auc_above_base_full",
]

env_vars = [
    "imperv_10m",
    "imperv_20m",
    "imperv_50m",
    "imperv_100m",
    "poll_no2_anmean",
    "poll_bc_anmean",
    "poll_pm25_anmean",
    "lst_temp_r100_y",
    "lst_temp_r50_y",
    "height",
    "insolation9",
]

CONTROL_VAR = "height"

# Cleaning
DROP_POLLUTION_NEGATIVE = True  # remove any rows where any poll_* < 0

# Outputs
OUTDIR = Path("/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/corr_outputs")
OUTDIR.mkdir(exist_ok=True)
SCATTER_DIR_ALL = OUTDIR / "scatterplots_all_pairs"
SCATTER_DIR_ALL.mkdir(exist_ok=True)
SCATTER_DIR_TOP = OUTDIR / "scatterplots_top_pairs_height_controlled"
SCATTER_DIR_TOP.mkdir(exist_ok=True)

# Heatmap settings
FIG_DPI = 300
HEATMAP_VLIM = 1.0  # color scale range [-1, 1]

# Significance settings
ANNOTATE_SIGNIFICANCE = True          # add *, **, *** in heatmap cells
MASK_NON_SIGNIFICANT = False          # set to True to blank out non-sig values in heatmap
ALPHA_SIG = 0.05                      # significance threshold for masking / stars
APPLY_FDR_BH = False                  # set True to apply FDR (Benjamini-Hochberg) per matrix before stars/masking

# Scatter settings
PLOT_ENV_ENV_ALL = True
PLOT_HEALTH_HEALTH_ALL = True
PLOT_HEALTH_ENV_ALL = False  # can be huge

ALPHA = 0.20
POINT_SIZE = 8
SMOOTH = True
SMOOTH_FRAC = 0.30

MAX_POINTS_TO_PLOT = 20000  # downsample only for plotting speed
RANDOM_SEED = 42

# Top-N pairs (selected from height-controlled matrices)
MAKE_TOP_PAIR_PLOTS = True
TOP_N_PAIRS = 10


# ----------------------------
# LOAD + CLEAN
# ----------------------------

df = pd.read_csv(DATA_PATH)

if DROP_POLLUTION_NEGATIVE:
    pollution_cols = [c for c in df.columns if c.startswith("poll_")]
    if pollution_cols:
        bad = df[pollution_cols].lt(0).any(axis=1)  # ignores NaNs
        removed = int(bad.sum())
        df = df.loc[~bad].copy()
        print(f"Removed {removed} rows where any poll_* < 0 across: {pollution_cols}")
    else:
        print("No poll_* columns found; skipping pollution<0 filter.")

# Ensure numeric for all variables of interest
all_vars = sorted(set(health_vars + env_vars + [CONTROL_VAR]))
missing = [c for c in all_vars if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in df: {missing}")

for c in all_vars:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ----------------------------
# STANDARDISE ALL VARIABLES (Z-SCORE)
# ----------------------------
def zscore_df(df_in: pd.DataFrame, cols: list[str], eps: float = 1e-12) -> pd.DataFrame:
    """
    Z-score standardisation for selected columns.
    Uses population SD (ddof=0). Columns with ~zero variance become 0.
    """
    df_out = df_in.copy()
    for c in cols:
        x = df_out[c]
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=0)
        if sd is None or np.isnan(sd) or sd < eps:
            df_out[c] = 0.0
        else:
            df_out[c] = (x - mu) / sd
    return df_out

df = zscore_df(df, all_vars)


# ----------------------------
# SIGNIFICANCE HELPERS
# ----------------------------

def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def fdr_bh_matrix(pmat: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction to a p-value matrix (cell-wise),
    ignoring NaNs. Returns a matrix of adjusted p-values with same shape/index/cols.
    """
    p = pmat.values.astype(float).ravel()
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return pmat.copy()

    _, p_adj, _, _ = multipletests(p[mask], method="fdr_bh")
    out = pmat.copy()
    out.values.ravel()[mask] = p_adj
    return out


# ----------------------------
# CORRELATION HELPERS
# ----------------------------

def spearman_scalar(x: pd.Series, y: pd.Series) -> float:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 3:
        return np.nan
    return float(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))

def spearman_with_p(x: pd.Series, y: pd.Series):
    """
    Returns (rho, pval, n) for Spearman correlation using the same row set.
    """
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    if n < 3:
        return np.nan, np.nan, n
    out = pg.corr(d.iloc[:, 0], d.iloc[:, 1], method="spearman")
    r = float(out["r"].iloc[0])
    p = float(out["p-val"].iloc[0])
    return r, p, n

def compute_spearman_matrix_with_p(df_in: pd.DataFrame, vars_a: list[str], vars_b: list[str]):
    """
    Returns (rho_matrix, pval_matrix, n_matrix)
    """
    rho = pd.DataFrame(index=vars_a, columns=vars_b, dtype=float)
    pval = pd.DataFrame(index=vars_a, columns=vars_b, dtype=float)
    nmat = pd.DataFrame(index=vars_a, columns=vars_b, dtype=float)

    for a in vars_a:
        for b in vars_b:
            r, p, n = spearman_with_p(df_in[a], df_in[b])
            rho.loc[a, b] = r
            pval.loc[a, b] = p
            nmat.loc[a, b] = n

    return rho, pval, nmat

def compute_partial_spearman_matrix_with_p(df_in: pd.DataFrame, vars_a: list[str], vars_b: list[str], covar: str):
    """
    Returns (rho_matrix, pval_matrix, n_matrix) for partial Spearman (control covar).
    """
    rho = pd.DataFrame(index=vars_a, columns=vars_b, dtype=float)
    pval = pd.DataFrame(index=vars_a, columns=vars_b, dtype=float)
    nmat = pd.DataFrame(index=vars_a, columns=vars_b, dtype=float)

    for a in vars_a:
        for b in vars_b:
            if a == covar or b == covar:
                rho.loc[a, b] = np.nan
                pval.loc[a, b] = np.nan
                nmat.loc[a, b] = np.nan
                continue

            d = df_in[[a, b, covar]].dropna()
            n = len(d)
            if n < 5:
                rho.loc[a, b] = np.nan
                pval.loc[a, b] = np.nan
                nmat.loc[a, b] = n
                continue

            try:
                pc = pg.partial_corr(data=d, x=a, y=b, covar=covar, method="spearman")
                rho.loc[a, b] = float(pc["r"].iloc[0])
                pval.loc[a, b] = float(pc["p-val"].iloc[0])
                nmat.loc[a, b] = n
            except Exception as e:
                print(f"Partial corr failed for {a} vs {b} | {covar}: {e}")
                rho.loc[a, b] = np.nan
                pval.loc[a, b] = np.nan
                nmat.loc[a, b] = n

    return rho, pval, nmat


def plot_corr_heatmap(mat: pd.DataFrame, pmat: pd.DataFrame | None, title: str, outfile: Path, vlim: float = 1.0):
    """
    Heatmap with optional significance stars based on pmat.
    If MASK_NON_SIGNIFICANT is True, cells with p >= ALPHA_SIG are masked (set to NaN).
    If APPLY_FDR_BH is True, p-values are adjusted per-matrix before stars/masking.
    """
    mat_plot = mat.copy()
    p_use = None

    if pmat is not None:
        p_use = pmat.copy()
        if APPLY_FDR_BH:
            p_use = fdr_bh_matrix(p_use)

        if MASK_NON_SIGNIFICANT:
            mat_plot = mat_plot.where(p_use < ALPHA_SIG)

    arr = mat_plot.values.astype(float)

    fig, ax = plt.subplots(
        figsize=(1 + 0.55 * mat_plot.shape[1], 1 + 0.45 * mat_plot.shape[0]),
        constrained_layout=True
    )
    im = ax.imshow(arr, vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title(title)

    ax.set_xticks(range(mat_plot.shape[1]))
    ax.set_xticklabels(mat_plot.columns, rotation=45, ha="right")
    ax.set_yticks(range(mat_plot.shape[0]))
    ax.set_yticklabels(mat_plot.index)

    # cell labels
    for i in range(mat_plot.shape[0]):
        for j in range(mat_plot.shape[1]):
            r = arr[i, j]
            if not np.isfinite(r):
                continue

            stars = ""
            if ANNOTATE_SIGNIFICANCE and (p_use is not None):
                p = float(p_use.values[i, j])
                stars = p_to_stars(p)

            ax.text(j, i, f"{r:.2f}{stars}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Spearman ρ (signed)")
    plt.savefig(outfile, dpi=FIG_DPI)
    plt.close()


def upper_triangle_pairs(mat_square: pd.DataFrame) -> pd.DataFrame:
    if mat_square.shape[0] != mat_square.shape[1]:
        raise ValueError("upper_triangle_pairs requires a square matrix.")
    vars_ = list(mat_square.index)
    rows = []
    for i in range(len(vars_)):
        for j in range(i + 1, len(vars_)):
            v1, v2 = vars_[i], vars_[j]
            rho = mat_square.loc[v1, v2]
            if np.isfinite(rho):
                rows.append((v1, v2, float(rho), abs(float(rho))))
    out = pd.DataFrame(rows, columns=["var1", "var2", "rho", "abs_rho"])
    return out.sort_values("abs_rho", ascending=False)


# ----------------------------
# SCATTERPLOT HELPERS
# ----------------------------

def safe_name(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace("/", "-")
         .replace("×", "x")
         .replace("(", "")
         .replace(")", "")
    )

def scatter_pair(df_in: pd.DataFrame, xcol: str, ycol: str, outpath: Path, title: str | None = None):
    d = df_in[[xcol, ycol]].dropna()
    n = len(d)
    if n < 3:
        return

    rho = spearman_scalar(d[xcol], d[ycol])

    # Downsample only for plotting speed (rho computed on full d)
    if n > MAX_POINTS_TO_PLOT:
        d_plot = d.sample(MAX_POINTS_TO_PLOT, random_state=RANDOM_SEED)
    else:
        d_plot = d

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(d_plot[xcol], d_plot[ycol], s=POINT_SIZE, alpha=ALPHA)

    if SMOOTH and len(d_plot) > 20:
        sm = lowess(d_plot[ycol], d_plot[xcol], frac=SMOOTH_FRAC, return_sorted=True)
        ax.plot(sm[:, 0], sm[:, 1])

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title if title else f"{ycol} vs {xcol}")

    ax.text(
        0.02, 0.98,
        f"Spearman ρ = {rho:.2f}\n n = {n}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
    )

    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close()

def run_all_pairs(pairs, subdir: Path):
    subdir.mkdir(exist_ok=True)
    total = 0
    for x, y in pairs:
        if x not in df.columns or y not in df.columns:
            continue
        out = subdir / f"{safe_name(y)}__vs__{safe_name(x)}.png"
        scatter_pair(df, x, y, out)
        total += 1
        if total % 50 == 0:
            print(f"{subdir.name}: generated {total} plots...")
    print(f"{subdir.name}: done ({total} plots)")


# ----------------------------
# 1) CORRELATION MATRICES (CSV + heatmaps + p-values)
# ----------------------------

print("Computing correlation matrices...")

# Health vs Environment (RAW)
raw_hxe, raw_hxe_p, raw_hxe_n = compute_spearman_matrix_with_p(df, health_vars, env_vars)
raw_hxe.to_csv(OUTDIR / "corr_spearman_signed_health_vs_env.csv")
raw_hxe_p.to_csv(OUTDIR / "pvals_spearman_health_vs_env.csv")
raw_hxe_n.to_csv(OUTDIR / "n_spearman_health_vs_env.csv")
plot_corr_heatmap(raw_hxe, raw_hxe_p, "Signed Spearman (health vs env) — raw", OUTDIR / "heatmap_raw_health_vs_env.png", vlim=HEATMAP_VLIM)

# Health vs Environment (PARTIAL)
par_hxe, par_hxe_p, par_hxe_n = compute_partial_spearman_matrix_with_p(df, health_vars, env_vars, CONTROL_VAR)
par_hxe.to_csv(OUTDIR / f"corr_partial_spearman_signed_health_vs_env_control_{CONTROL_VAR}.csv")
par_hxe_p.to_csv(OUTDIR / f"pvals_partial_spearman_health_vs_env_control_{CONTROL_VAR}.csv")
par_hxe_n.to_csv(OUTDIR / f"n_partial_spearman_health_vs_env_control_{CONTROL_VAR}.csv")
plot_corr_heatmap(par_hxe, par_hxe_p, f"Signed partial Spearman (health vs env) — control {CONTROL_VAR}", OUTDIR / f"heatmap_partial_health_vs_env_control_{CONTROL_VAR}.png", vlim=HEATMAP_VLIM)

# Env vs Env (RAW)
raw_env, raw_env_p, raw_env_n = compute_spearman_matrix_with_p(df, env_vars, env_vars)
np.fill_diagonal(raw_env.values, np.nan)
np.fill_diagonal(raw_env_p.values, np.nan)
np.fill_diagonal(raw_env_n.values, np.nan)
raw_env.to_csv(OUTDIR / "corr_spearman_signed_env_vs_env.csv")
raw_env_p.to_csv(OUTDIR / "pvals_spearman_env_vs_env.csv")
raw_env_n.to_csv(OUTDIR / "n_spearman_env_vs_env.csv")
plot_corr_heatmap(raw_env, raw_env_p, "Signed Spearman (env vs env) — raw", OUTDIR / "heatmap_raw_env_vs_env.png", vlim=HEATMAP_VLIM)

# Env vs Env (PARTIAL)
par_env, par_env_p, par_env_n = compute_partial_spearman_matrix_with_p(df, env_vars, env_vars, CONTROL_VAR)
np.fill_diagonal(par_env.values, np.nan)
np.fill_diagonal(par_env_p.values, np.nan)
np.fill_diagonal(par_env_n.values, np.nan)
par_env.to_csv(OUTDIR / f"corr_partial_spearman_signed_env_vs_env_control_{CONTROL_VAR}.csv")
par_env_p.to_csv(OUTDIR / f"pvals_partial_spearman_env_vs_env_control_{CONTROL_VAR}.csv")
par_env_n.to_csv(OUTDIR / f"n_partial_spearman_env_vs_env_control_{CONTROL_VAR}.csv")
plot_corr_heatmap(par_env, par_env_p, f"Signed partial Spearman (env vs env) — control {CONTROL_VAR}", OUTDIR / f"heatmap_partial_env_vs_env_control_{CONTROL_VAR}.png", vlim=HEATMAP_VLIM)

# Health vs Health (RAW)
raw_health, raw_health_p, raw_health_n = compute_spearman_matrix_with_p(df, health_vars, health_vars)
np.fill_diagonal(raw_health.values, np.nan)
np.fill_diagonal(raw_health_p.values, np.nan)
np.fill_diagonal(raw_health_n.values, np.nan)
raw_health.to_csv(OUTDIR / "corr_spearman_signed_health_vs_health.csv")
raw_health_p.to_csv(OUTDIR / "pvals_spearman_health_vs_health.csv")
raw_health_n.to_csv(OUTDIR / "n_spearman_health_vs_health.csv")
plot_corr_heatmap(raw_health, raw_health_p, "Signed Spearman (health vs health) — raw", OUTDIR / "heatmap_raw_health_vs_health.png", vlim=HEATMAP_VLIM)

# Health vs Health (PARTIAL)
par_health, par_health_p, par_health_n = compute_partial_spearman_matrix_with_p(df, health_vars, health_vars, CONTROL_VAR)
np.fill_diagonal(par_health.values, np.nan)
np.fill_diagonal(par_health_p.values, np.nan)
np.fill_diagonal(par_health_n.values, np.nan)
par_health.to_csv(OUTDIR / f"corr_partial_spearman_signed_health_vs_health_control_{CONTROL_VAR}.csv")
par_health_p.to_csv(OUTDIR / f"pvals_partial_spearman_health_vs_health_control_{CONTROL_VAR}.csv")
par_health_n.to_csv(OUTDIR / f"n_partial_spearman_health_vs_health_control_{CONTROL_VAR}.csv")
plot_corr_heatmap(par_health, par_health_p, f"Signed partial Spearman (health vs health) — control {CONTROL_VAR}", OUTDIR / f"heatmap_partial_health_vs_health_control_{CONTROL_VAR}.png", vlim=HEATMAP_VLIM)

print("Correlation matrices saved to:", OUTDIR.resolve())


# ----------------------------
# 2) SCATTERPLOTS FOR ALL PAIRS (robust)
# ----------------------------

print("Generating scatterplots for all pairs...")

if PLOT_ENV_ENV_ALL:
    env_pairs = list(combinations(env_vars, 2))
    run_all_pairs(env_pairs, SCATTER_DIR_ALL / "env_vs_env")

if PLOT_HEALTH_HEALTH_ALL:
    health_pairs = list(combinations(health_vars, 2))
    run_all_pairs(health_pairs, SCATTER_DIR_ALL / "health_vs_health")

if PLOT_HEALTH_ENV_ALL:
    cross_pairs = list(product(env_vars, health_vars))  # x=env, y=health
    run_all_pairs(cross_pairs, SCATTER_DIR_ALL / "health_vs_env")


# ----------------------------
# 3) SCATTERPLOTS FOR TOP-N PAIRS (selected from HEIGHT-CONTROLLED matrices)
# ----------------------------

if MAKE_TOP_PAIR_PLOTS:
    print(f"Generating top {TOP_N_PAIRS} pair scatterplots selected from height-controlled matrices...")

    # Env top pairs from partial matrix
    env_top = upper_triangle_pairs(par_env).head(TOP_N_PAIRS)
    env_top.to_csv(SCATTER_DIR_TOP / f"top{TOP_N_PAIRS}_env_pairs_partial_control_{CONTROL_VAR}.csv", index=False)

    env_dir = SCATTER_DIR_TOP / "env_vs_env_top"
    env_dir.mkdir(exist_ok=True)
    for k, row in env_top.reset_index(drop=True).iterrows():
        v1, v2 = row["var1"], row["var2"]
        out = env_dir / f"env_pair_{k+1:02d}_{safe_name(v1)}__{safe_name(v2)}.png"
        scatter_pair(df, v1, v2, out, title=f"ENV (partial control {CONTROL_VAR}) #{k+1}: {v2} vs {v1}")

    # Health top pairs from partial matrix
    health_top = upper_triangle_pairs(par_health).head(TOP_N_PAIRS)
    health_top.to_csv(SCATTER_DIR_TOP / f"top{TOP_N_PAIRS}_health_pairs_partial_control_{CONTROL_VAR}.csv", index=False)

    health_dir = SCATTER_DIR_TOP / "health_vs_health_top"
    health_dir.mkdir(exist_ok=True)
    for k, row in health_top.reset_index(drop=True).iterrows():
        v1, v2 = row["var1"], row["var2"]
        out = health_dir / f"health_pair_{k+1:02d}_{safe_name(v1)}__{safe_name(v2)}.png"
        scatter_pair(df, v1, v2, out, title=f"HEALTH (partial control {CONTROL_VAR}) #{k+1}: {v2} vs {v1}")

print("Done.")
