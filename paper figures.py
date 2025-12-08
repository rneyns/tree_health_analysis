#!/usr/bin/env python3
"""
Part 1: Boxplots of RMSE of the double-logistic NDVI curve fit,
        one box per tree species.

Part 2: One big figure with:
    - For each species: a horizontal barplot of the top 5 environmental
      variables whose values are most strongly correlated with RMSE
      (Pearson, by |r|).
    - Bottom subplot: for each species, a single bar showing the
      environmental variable with the largest |r| with RMSE and
      the value of that correlation.

Assumes each species has its own folder with:
    ndvi_metrics_clean.csv

and that this file contains at least:
    - 'rmse'          : fit error
    - environmental columns such as:
        'poll_*', 'impervious_r*', 'temp_r*', 'lst_temp_r*', 'insolation*'
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

# ------------------ USER CONFIG ------------------

# Map "nice species name" -> path to its ndvi_metrics_clean.csv
SPECIES_FILES = {
    "Aesculus hippocastanum": (
        "/Users/robbe_neyns/Documents/Work_local/research/"
        "UHI tree health/Data analysis/Data/PlanetScope/"
        "aesculus hippocastanum/ndvi_metrics_clean.csv"
    ),
    "Acer pseudoplatanus": (
        "/Users/robbe_neyns/Documents/Work_local/research/"
        "UHI tree health/Data analysis/Data/PlanetScope/"
        "acer pseudoplatanus/ndvi_metrics_clean.csv"
    ),
    "Platanus × acerifolia": (
'/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Platanus x acerifolia/ndvi_metrics_clean.csv'
    ),
    "Acer platanoides": (
        "/Users/robbe_neyns/Documents/Work_local/research/"
        "UHI tree health/Data analysis/Data/PlanetScope/"
        "Acer platanoides/ndvi_metrics_clean.csv"
    ),
    "Tilia x euchlora": (
        "/Users/robbe_neyns/Documents/Work_local/research/"
        "UHI tree health/Data analysis/Data/PlanetScope/"
        "tilia x euchlora/ndvi_metrics_clean.csv"
    ),
}


# Where to save the figures
OUTPUT_FIG_RMSE_BOX = (
    "/Users/robbe_neyns/Documents/Work_local/research/"
    "UHI tree health/Figures paper/rmse_boxplot_by_species.png"
)

OUTPUT_FIG_RMSE_ENV = (
    "/Users/robbe_neyns/Documents/Work_local/research/"
    "UHI tree health/Figures paper/rmse_env_correlations_by_species.png"
)

# Correlation x-axis limits
CORR_XMIN = -0.18
CORR_XMAX =  0.13

# -------------------------------------------------


def pretty_env_name(col: str) -> str:
    """
    Turn internal column names into paper-friendly labels.
    Examples:
      'impervious_r50'   -> 'Impervious surface (50 m)'
      'temp_r200'        -> 'Air temperature (200 m)'
      'lst_temp_r100'    -> 'Land surface temperature (100 m)'
      'insolation9'      -> 'Insolation (9 m)'
      'poll_NO2'         -> 'Pollution NO2'
    """
    if col.startswith("impervious_r"):
        m = re.search(r"impervious_r(\d+)", col)
        r = m.group(1) if m else "?"
        return f"Impervious surface ({r} m)"

    if col.startswith("temp_r"):
        m = re.search(r"temp_r(\d+)", col)
        r = m.group(1) if m else "?"
        return f"Air temperature ({r} m)"

    if col.startswith("lst_temp_r"):
        m = re.search(r"lst_temp_r(\d+)", col)
        r = m.group(1) if m else "?"
        return f"Land surface temperature ({r} m)"

    if col.startswith("insolation"):
        m = re.search(r"insolation(\d+)", col)
        r = m.group(1) if m else "?"
        return f"Insolation ({r} m)"

    if col.startswith("poll_"):
        pollutant = col[5:].upper().replace("_", " ")
        return f"Pollution {pollutant}"

    return col.replace("_", " ")


def main():
    species_names = []
    rmse_data = []
    species_counts = {}  # {species: n_samples}
    total_samples = 0

    # For part 2: keep full rows (with env vars) for all species
    all_metrics_rows = []

    # --------- Load RMSE data for each species ---------
    for species, path in SPECIES_FILES.items():
        print(f"\n[INFO] Processing species: {species}")
        print(f"       CSV path: {path}")

        if not os.path.exists(path):
            print(f"[WARN] File not found for {species}: {path}")
            continue

        df = pd.read_csv(path)
        print(f"[INFO] Columns in {species} CSV: {list(df.columns)}")
        print(f"[INFO] Number of rows in {species} CSV: {len(df)}")

        if "rmse" not in df.columns:
            print(f"[WARN] 'rmse' column not found in {path}; skipping {species}")
            continue

        df_valid = df[df["rmse"].notna()].copy()
        if df_valid.empty:
            print(f"[WARN] No valid RMSE values for {species}; skipping.")
            continue

        rmse_values = df_valid["rmse"]
        n = len(rmse_values)

        species_names.append(species)
        rmse_data.append(rmse_values.values)
        species_counts[species] = n
        total_samples += n

        df_valid["species"] = species
        all_metrics_rows.append(df_valid)

        print(f"[OK] Loaded {n} RMSE values for {species}")

    if not rmse_data:
        print("[ERROR] No RMSE data found for any species. Check paths and column names.")
        return

    # =====================================================
    # PART 1: Boxplot of RMSE by species
    # =====================================================
    print("\n[INFO] Creating RMSE boxplot by species...")

    plt.figure(figsize=(8, 5))

    bp = plt.boxplot(
        rmse_data,
        patch_artist=True,
        showfliers=True,
    )

    labels_with_n = [f"{sp} (n={species_counts[sp]})" for sp in species_names]
    plt.xticks(
        range(1, len(species_names) + 1),
        labels_with_n,
        rotation=20,
        ha="right"
    )

    for box in bp["boxes"]:
        box.set_alpha(0.7)

    plt.ylabel("RMSE of NDVI double-logistic fit")
    plt.xlabel("Tree species")
    plt.grid(axis="y", alpha=0.3)

    ax = plt.gca()
    ax.text(
        0.98, 0.02,
        f"Total samples: n = {total_samples}",
        ha="right", va="bottom",
        transform=ax.transAxes,
        fontsize=10, alpha=0.8,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_FIG_RMSE_BOX), exist_ok=True)
    plt.savefig(OUTPUT_FIG_RMSE_BOX, dpi=300)
    plt.close()

    print("\n----- SUMMARY: RMSE by species -----")
    print(f"Total RMSE samples used: {total_samples}")
    for sp, n in species_counts.items():
        print(f"  {sp}: {n}")
    print(f"[OK] Saved RMSE boxplot to: {OUTPUT_FIG_RMSE_BOX}")

    # =====================================================
    # PART 2: Per-species correlations + bottom summary
    # =====================================================
    print("\n[INFO] Computing per-species correlations between RMSE and environmental variables...")

    if not all_metrics_rows:
        print("[ERROR] No metrics rows collected; cannot compute correlations.")
        return

    all_df = pd.concat(all_metrics_rows, ignore_index=True)

    env_prefixes = ("poll_", "impervious_r", "temp_r", "lst_temp_r", "insolation")
    env_cols = [
        c for c in all_df.columns
        if any(c.startswith(p) for p in env_prefixes) and is_numeric_dtype(all_df[c])
    ]

    if not env_cols:
        print("[WARN] No environmental columns found matching expected patterns.")
        return

    n_species = len(species_names)
    if n_species == 0:
        print("[ERROR] No species with valid RMSE data; cannot plot correlations.")
        return

    # We want one subplot per species + 1 summary subplot at bottom
    n_rows = n_species + 1
    n_cols = 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10, 2.6 * n_rows),
        sharex=True
    )
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # For the bottom summary panel
    best_r_by_species = {}
    best_col_by_species = {}

    print("\n----- TOP 5 ENVIRONMENTAL CORRELATES WITH RMSE (PER SPECIES) -----")

    # Species panels
    for idx, species in enumerate(species_names):
        ax = axes[idx]
        df_sp = all_df[all_df["species"] == species].copy()
        if df_sp.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(species)
            continue

        corr_results = []
        for col in env_cols:
            sub = df_sp[["rmse", col]].dropna()
            if len(sub) < 2:
                continue
            r = sub["rmse"].corr(sub[col], method="pearson")
            if pd.isna(r):
                continue
            corr_results.append((col, r))

        if not corr_results:
            ax.text(0.5, 0.5, "No valid correlations", ha="center", va="center")
            ax.set_title(species)
            continue

        corr_results.sort(key=lambda x: abs(x[1]), reverse=True)
        top_corrs = corr_results[:5]

        # Store best correlation for the summary panel
        best_col, best_r = top_corrs[0]
        best_r_by_species[species] = best_r
        best_col_by_species[species] = best_col

        print(f"\nSpecies: {species}")
        for col, r in top_corrs:
            print(f"  {col}: r = {r:.3f}")

        var_cols = [c for c, _ in top_corrs]
        var_names_pretty = [pretty_env_name(c) for c in var_cols]
        r_values = [r for _, r in top_corrs]

        y_positions = range(len(var_names_pretty))
        colors = ["tab:red" if r > 0 else "tab:blue" for r in r_values]

        ax.barh(y_positions, r_values, color=colors)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(var_names_pretty, fontsize=9)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(species, fontsize=11)

        ax.set_xlim(CORR_XMIN, CORR_XMAX)

        for y, r in zip(y_positions, r_values):
            text_x = r + (0.01 if r >= 0 else -0.01)
            text_x = max(CORR_XMIN + 0.005, min(CORR_XMAX - 0.005, text_x))
            ax.text(
                text_x,
                y,
                f"{r:.2f}",
                va="center",
                ha="left" if r >= 0 else "right",
                fontsize=8
            )

    # Bottom summary panel: one bar per species = best correlation
    ax_sum = axes[-1]
    y_positions = range(len(species_names))
    summary_r = [best_r_by_species.get(sp, 0.0) for sp in species_names]
    summary_cols = [best_col_by_species.get(sp, None) for sp in species_names]
    colors = ["tab:red" if r > 0 else "tab:blue" for r in summary_r]

    ax_sum.barh(y_positions, summary_r, color=colors)
    ax_sum.axvline(0, color="black", linewidth=1)
    ax_sum.set_xlim(CORR_XMIN, CORR_XMAX)
    ax_sum.set_yticks(y_positions)
    ax_sum.set_yticklabels(species_names, fontsize=9)
    ax_sum.set_title("Strongest RMSE–environment correlation per species", fontsize=11)

    '''
    # Annotate with r and variable name
    for y, sp, r, col in zip(y_positions, species_names, summary_r, summary_cols):
        if col is None:
            continue
        label = f"{pretty_env_name(col)} (r={r:.2f})"
        text_x = r + (0.01 if r >= 0 else -0.01)
        text_x = max(CORR_XMIN + 0.005, min(CORR_XMAX - 0.005, text_x))
        ax_sum.text(
            text_x,
            y,
            label,
            va="center",
            ha="left" if r >= 0 else "right",
            fontsize=8
        )
    '''

    fig.text(0.5, 0.03, "Pearson correlation with RMSE", ha="center")

    plt.tight_layout(rect=[0.05, 0.06, 0.98, 0.98])
    os.makedirs(os.path.dirname(OUTPUT_FIG_RMSE_ENV), exist_ok=True)
    plt.savefig(OUTPUT_FIG_RMSE_ENV, dpi=300)
    plt.close()

    print(f"\n[OK] Saved per-species RMSE–environment correlations figure to: {OUTPUT_FIG_RMSE_ENV}")


if __name__ == "__main__":
    main()
