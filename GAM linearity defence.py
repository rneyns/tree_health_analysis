#!/usr/bin/env python3
"""
GAM partial effects vs linear fit — linearity defense appendix figures.

For each health metric, produces ONE figure with:
    rows = species (5)
    cols = predictors (4)

Each panel shows:
    - Partial response curve from a GAM (controlling for other predictors + height)
    - Linear partial regression line (same controls)
    - Rug plot of observed predictor values
    - 95% confidence interval for the GAM smooth

This allows visual comparison of whether the linear approximation captures
the directional pattern of the GAM smooth, justifying the use of MLR
as a linear approximation even where the RESET test flagged misspecification.

Dependencies:
    pip install numpy pandas matplotlib scikit-learn pygam scipy
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from pygam import LinearGAM, s

# ─────────────────────────────────────────────
# USER SETTINGS  ←  edit these paths
# ─────────────────────────────────────────────

SPECIES_CSVS = {
    "Acer platanoides":      "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer platanoides/ndvi_metrics_with_impervious.csv",
    "Acer pseudoplatanus":   "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/acer pseudoplatanus/ndvi_metrics_with_impervious.csv",
    "Aesculus hippocastanum":"/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/aesculus hippocastanum/ndvi_metrics_with_impervious.csv",
    "Platanus × acerifolia": "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/platanus x acerifolia/ndvi_metrics_with_impervious.csv",
    "Tilia × euchlora":      "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/tilia x euchlora/ndvi_metrics_with_impervious.csv",
}

OUT_DIR = Path("/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/ndvi background investigations/_GAM_LINEARITY_DEFENSE")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# VARIABLE DEFINITIONS
# ─────────────────────────────────────────────

HEALTH_VARS = {
    "ndvi_peak":            "Peak NDVI",
    "sos_doy":              "Start of season (DOY)",
    "los_days":             "Length of season (days)",
    "amplitude":            "NDVI amplitude",
    "auc_above_base_full":  "Seasonal NDVI integral",
}

PREDICTORS = {
    "imperv_10m":       "Imperviousness",
    "poll_bc_anmean":   "Black carbon",
    "lst_temp_r50_y":   "LST",
    "insolation9":      "Solar radiation",
}

CONTROL_VAR = "height"

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────

SPECIES_COLORS = {
    "Acer platanoides":       "#4C9BE8",
    "Acer pseudoplatanus":    "#E8854C",
    "Aesculus hippocastanum": "#5DBB63",
    "Platanus × acerifolia":  "#B05DBB",
    "Tilia × euchlora":       "#BB9A5D",
}

N_SPLINES   = 10
LAM_GRID    = np.logspace(-3, 3, 13)
N_GRID      = 200   # points along predictor range for smooth curve
RUG_ALPHA   = 0.15
RUG_HEIGHT  = 0.03
CI_ALPHA    = 0.15
FIG_DPI     = 240

BIOLOGICAL_FILTERS = {
    "los_days":             lambda x: (x >= 60)  & (x <= 365),
    "sos_doy":              lambda x: (x >= 1)   & (x <= 250),
    "ndvi_peak":            lambda x: (x >= -0.1) & (x <= 1.0),
    "amplitude":            lambda x: (x >= -0.1) & (x <= 1.0),
    "auc_above_base_full":  lambda x: x > -1e9,
    "height":               lambda x: x > 1,
    "poll_bc_anmean":       lambda x: x > 0,
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / sd if sd > 1e-12 else s * 0.0


def load_species(csv_path: str, health: str, predictors: list, control: str) -> pd.DataFrame | None:
    """Load, filter, and standardise data for one species × health metric."""
    df = pd.read_csv(csv_path)
    needed = [health, control] + predictors
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"  [skip] missing columns: {missing}")
        return None

    df = df[needed].copy()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    # biological filters
    for c in needed:
        if c in BIOLOGICAL_FILTERS:
            df = df[BIOLOGICAL_FILTERS[c](df[c])]

    if len(df) < 80:
        print(f"  [skip] n={len(df)} too small")
        return None

    # standardise everything
    for c in needed:
        df[c] = zscore(df[c])

    return df.reset_index(drop=True)


def partial_gam_curve(df: pd.DataFrame, focal: str, others: list, health: str):
    """
    Fit a GAM(focal + others) predicting health.
    Return (x_grid, y_smooth, y_lower, y_upper) evaluated along focal,
    with all other predictors held at their mean (0 after standardisation).
    """
    all_preds = [focal] + others
    X = df[all_preds].values
    y = df[health].values

    # build terms: one smooth per predictor
    terms = s(0, n_splines=N_SPLINES)
    for j in range(1, len(all_preds)):
        terms = terms + s(j, n_splines=N_SPLINES)

    gam = LinearGAM(terms)
    gam.gridsearch(X, y, lam=LAM_GRID, progress=False)

    # grid along focal predictor; others held at 0 (standardised mean)
    x_min, x_max = df[focal].quantile(0.02), df[focal].quantile(0.98)
    x_grid = np.linspace(x_min, x_max, N_GRID)

    X_pred = np.zeros((N_GRID, len(all_preds)))
    X_pred[:, 0] = x_grid   # focal varies
    # others stay at 0 (mean after standardisation)

    y_smooth = gam.predict(X_pred)
    ci = gam.confidence_intervals(X_pred, width=0.95)
    y_lower, y_upper = ci[:, 0], ci[:, 1]

    return x_grid, y_smooth, y_lower, y_upper


def partial_linear_line(df: pd.DataFrame, focal: str, others: list, health: str):
    """
    OLS partial regression line for focal predictor, controlling for others.
    Returns (x_grid, y_line) centred at mean of y.
    """
    all_preds = [focal] + others
    X = df[all_preds].values
    y = df[health].values

    # add intercept
    Xb = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)

    x_min, x_max = df[focal].quantile(0.02), df[focal].quantile(0.98)
    x_grid = np.linspace(x_min, x_max, N_GRID)

    # others at 0 (standardised mean), intercept included
    y_line = coef[0] + coef[1] * x_grid   # coef[1] is the focal coefficient

    return x_grid, y_line


# ─────────────────────────────────────────────
# MAIN PLOTTING LOOP — one figure per health metric
# ─────────────────────────────────────────────

def make_figure(health_col: str, health_label: str):
    species_list  = list(SPECIES_CSVS.keys())
    pred_cols     = list(PREDICTORS.keys())
    pred_labels   = list(PREDICTORS.values())
    n_species     = len(species_list)
    n_preds       = len(pred_cols)

    fig = plt.figure(figsize=(3.5 * n_preds, 2.8 * n_species))
    gs  = gridspec.GridSpec(
        n_species, n_preds,
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.97, top=0.93, bottom=0.06
    )

    fig.suptitle(
        f"GAM partial effects vs linear fit — {health_label}",
        fontsize=13, fontweight="bold", y=0.975
    )

    # column headers (predictor names)
    for j, plabel in enumerate(pred_labels):
        ax_top = fig.add_subplot(gs[0, j])
        ax_top.set_title(plabel, fontsize=10, fontweight="bold", pad=6)

    for i, (sp_name, csv_path) in enumerate(species_list):
        others = [p for p in pred_cols]  # will remove focal inside loop

        df = load_species(csv_path, health_col, pred_cols, CONTROL_VAR)
        color = SPECIES_COLORS[sp_name]

        for j, focal in enumerate(pred_cols):
            ax = fig.add_subplot(gs[i, j])

            # row label: species name (italic) on leftmost column only
            if j == 0:
                ax.set_ylabel(f"$\\it{{{sp_name.replace(' ', '\\ ')}}}$",
                              fontsize=8, labelpad=4)

            if df is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="grey")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            other_preds = [p for p in pred_cols if p != focal] + [CONTROL_VAR]

            # ── GAM smooth ──
            try:
                xg, ys, yl, yu = partial_gam_curve(df, focal, other_preds, health_col)
                ax.fill_between(xg, yl, yu, alpha=CI_ALPHA, color=color)
                ax.plot(xg, ys, color=color, linewidth=2.0, label="GAM")
            except Exception as e:
                print(f"  [warn] GAM failed for {sp_name} / {focal}: {e}")

            # ── Linear partial regression ──
            try:
                xg_lin, y_lin = partial_linear_line(df, focal, other_preds, health_col)
                ax.plot(xg_lin, y_lin, color="black", linewidth=1.2,
                        linestyle="--", label="Linear")
            except Exception as e:
                print(f"  [warn] Linear failed for {sp_name} / {focal}: {e}")

            # ── Rug ──
            x_vals = df[focal].values
            y_range = ax.get_ylim()
            rug_y = y_range[0] + RUG_HEIGHT * (y_range[1] - y_range[0])
            ax.plot(x_vals, np.full_like(x_vals, rug_y), "|",
                    color=color, alpha=RUG_ALPHA, markersize=4)

            # ── Reference line at y=0 ──
            ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")

            ax.tick_params(labelsize=7)
            ax.set_xlabel("Standardised predictor", fontsize=7)

            # only show y-axis label on leftmost panels
            if j != 0:
                ax.set_yticklabels([])

    # ── Legend (bottom right, outside grid) ──
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="grey",  linewidth=2.0, label="GAM smooth"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Linear fit"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, 0.005))

    out_path = OUT_DIR / f"gam_vs_linear_{health_col}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Convert dict to list of tuples once so we iterate consistently
    species_items = list(SPECIES_CSVS.items())

    # Monkey-patch make_figure to use species_items (cleaner than global)
    import types

    def make_figure_fixed(health_col: str, health_label: str):
        pred_cols   = list(PREDICTORS.keys())
        pred_labels = list(PREDICTORS.values())
        n_species   = len(species_items)
        n_preds     = len(pred_cols)

        fig = plt.figure(figsize=(3.5 * n_preds, 2.8 * n_species))
        gs  = gridspec.GridSpec(
            n_species, n_preds,
            hspace=0.50, wspace=0.35,
            left=0.09, right=0.97, top=0.92, bottom=0.07
        )

        fig.suptitle(
            f"GAM partial effects vs linear fit — {health_label}",
            fontsize=13, fontweight="bold", y=0.975
        )

        for j, plabel in enumerate(pred_labels):
            # invisible axis just for the column title
            ax_hdr = fig.add_subplot(gs[0, j])
            ax_hdr.set_title(plabel, fontsize=10, fontweight="bold", pad=8)

        for i, (sp_name, csv_path) in enumerate(species_items):
            print(f"  {sp_name} / {health_label} ...")
            color = SPECIES_COLORS[sp_name]

            df = load_species(csv_path, health_col, pred_cols, CONTROL_VAR)

            for j, focal in enumerate(pred_cols):
                ax = fig.add_subplot(gs[i, j])

                if j == 0:
                    # italic species name as y-axis label
                    parts = sp_name.split(" ")
                    italic = "$\\it{" + "\\ ".join(parts) + "}$"
                    ax.set_ylabel(italic, fontsize=8, labelpad=4)

                if df is None:
                    ax.text(0.5, 0.5, "insufficient data",
                            ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="grey")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                other_preds = [p for p in pred_cols if p != focal] + [CONTROL_VAR]

                # ── GAM smooth + CI ──
                gam_ok = False
                try:
                    xg, ys, yl, yu = partial_gam_curve(df, focal, other_preds, health_col)
                    ax.fill_between(xg, yl, yu, alpha=CI_ALPHA, color=color, zorder=1)
                    ax.plot(xg, ys, color=color, linewidth=2.0, zorder=3)
                    gam_ok = True
                except Exception as e:
                    print(f"    [warn] GAM failed: {e}")

                # ── Linear partial regression ──
                try:
                    xg_lin, y_lin = partial_linear_line(df, focal, other_preds, health_col)
                    ax.plot(xg_lin, y_lin, color="black", linewidth=1.4,
                            linestyle="--", zorder=4)
                except Exception as e:
                    print(f"    [warn] Linear failed: {e}")

                # ── Rug ──
                x_obs = df[focal].values
                ylims  = ax.get_ylim()
                if ylims[1] > ylims[0]:
                    rug_y = ylims[0] + RUG_HEIGHT * (ylims[1] - ylims[0])
                else:
                    rug_y = 0
                ax.plot(x_obs, np.full_like(x_obs, rug_y), "|",
                        color=color, alpha=RUG_ALPHA, markersize=3, zorder=2)

                # ── Zero reference ──
                ax.axhline(0, color="grey", linewidth=0.5, linestyle=":", zorder=0)

                ax.tick_params(labelsize=7)
                ax.set_xlabel("Std. predictor value", fontsize=7)
                if j != 0:
                    ax.set_yticklabels([])

        # shared legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="steelblue", linewidth=2.0, label="GAM smooth (± 95% CI)"),
            Line2D([0], [0], color="black",     linewidth=1.4, linestyle="--", label="Linear partial fit"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=2, fontsize=9, frameon=True,
                   bbox_to_anchor=(0.5, 0.002))

        out_path = OUT_DIR / f"gam_vs_linear_{health_col}.png"
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  → saved: {out_path}\n")

    print("=" * 60)
    print("GAM linearity defense — generating figures")
    print("=" * 60)

    for health_col, health_label in HEALTH_VARS.items():
        print(f"\n[{health_label}]")
        make_figure_fixed(health_col, health_label)

    print("\nDone. All figures saved to:", OUT_DIR)