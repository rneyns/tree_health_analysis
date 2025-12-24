import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILES = {
    "Acer platanoides": "uhi_pollution_treehealth_summary_acer_platanoides.csv",
    "Acer pseudoplatanus": "uhi_pollution_treehealth_summary_acer_pseudoplatanus.csv",
    "Aesculus hippocastanum": "uhi_pollution_treehealth_summary_aesculus_hippocastanum.csv",
    "Platanus × acerifolia": "uhi_pollution_treehealth_summary_platanus_x_acerifolia.csv",
    "Tilia × euchlora": "uhi_pollution_treehealth_summary_tilia_x_euchlora.csv",
}

METRIC_ORDER = [
    "amplitude",
    "auc_above_base_full",
    "slope_sos_peak",
    "sos_doy",
    "peak_doy",
    "eos_doy",
    "los_days",
]

PRED_ORDER = ["lst_temp_r50", "lst_temp_r100", "poll_no2_anmean"]

def load_all(files: dict) -> pd.DataFrame:
    frames = []
    for species, path in files.items():
        df = pd.read_csv(path)
        df["species"] = species
        df["sig_spear"] = df["partial_spearman_p_fdr"] < 0.05
        df["sig_gam"] = df["gam_p_term0_fdr_approx"] < 0.05
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["health_metric"] = pd.Categorical(out["health_metric"], METRIC_ORDER, ordered=True)
    out["predictor"] = pd.Categorical(out["predictor"], PRED_ORDER, ordered=True)
    return out

df = load_all(FILES)

# -------------------------
# Figure 1: Heatmap per species (rho) with significance markers
# -------------------------
def plot_species_heatmaps(df: pd.DataFrame, outfile: str | None = None):
    species_list = list(df["species"].unique())
    nsp = len(species_list)

    fig, axes = plt.subplots(nrows=1, ncols=nsp, figsize=(4*nsp, 6), constrained_layout=True)
    if nsp == 1:
        axes = [axes]

    for ax, sp in zip(axes, species_list):
        d = df[df["species"] == sp].copy()
        pivot = d.pivot(index="health_metric", columns="predictor", values="partial_spearman_rho").loc[METRIC_ORDER, PRED_ORDER]
        im = ax.imshow(pivot.values, aspect="auto")

        # Add significance markers:
        # - "*" if Spearman FDR significant
        # - "•" if GAM FDR significant (and not Spearman)
        d_idx = d.set_index(["health_metric", "predictor"])
        for i, m in enumerate(METRIC_ORDER):
            for j, p in enumerate(PRED_ORDER):
                if (m, p) not in d_idx.index:
                    continue
                row = d_idx.loc[(m, p)]
                txt = ""
                if bool(row["sig_spear"]):
                    txt = "*"
                elif bool(row["sig_gam"]):
                    txt = "•"
                if txt:
                    ax.text(j, i, txt, ha="center", va="center")

        ax.set_title(sp)
        ax.set_xticks(range(len(PRED_ORDER)))
        ax.set_xticklabels(PRED_ORDER, rotation=45, ha="right")
        ax.set_yticks(range(len(METRIC_ORDER)))
        ax.set_yticklabels(METRIC_ORDER)

    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label("Partial Spearman ρ (FDR markers: * Spearman, • GAM)")

    if outfile:
        plt.savefig(outfile, dpi=300)
    plt.show()

plot_species_heatmaps(df, outfile="fig1_species_heatmaps.png")

# -------------------------
# Figure 2: “Fingerprint” dot plot (all species stacked)
# -------------------------
def plot_fingerprint(df: pd.DataFrame, predictor: str, outfile: str | None = None):
    d = df[df["predictor"] == predictor].copy()
    species_list = list(d["species"].unique())

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # y positions by metric; x is rho; species gets small offsets
    y_base = {m:i for i,m in enumerate(METRIC_ORDER)}
    offsets = np.linspace(-0.25, 0.25, len(species_list))

    for off, sp in zip(offsets, species_list):
        ds = d[d["species"] == sp].sort_values("health_metric")
        y = np.array([y_base[m] for m in ds["health_metric"]]) + off
        x = ds["partial_spearman_rho"].values

        # marker filled if Spearman sig, hollow if only GAM sig, tiny if none
        for xi, yi, ss, sg in zip(x, y, ds["sig_spear"], ds["sig_gam"]):
            if ss:
                ax.plot(xi, yi, marker="o", markersize=7, linestyle="None")
            elif sg:
                ax.plot(xi, yi, marker="o", markersize=7, linestyle="None", fillstyle="none")
            else:
                ax.plot(xi, yi, marker=".", markersize=4, linestyle="None")

    ax.axvline(0, linewidth=1)
    ax.set_yticks(range(len(METRIC_ORDER)))
    ax.set_yticklabels(METRIC_ORDER)
    ax.set_xlabel("Partial Spearman ρ")
    ax.set_title(f"Species fingerprints for {predictor}\n(filled = Spearman FDR<0.05; open = GAM FDR<0.05 only)")

    if outfile:
        plt.savefig(outfile, dpi=300)
    plt.show()

plot_fingerprint(df, predictor="lst_temp_r100", outfile="fig2_fingerprint_lst100.png")
plot_fingerprint(df, predictor="poll_no2_anmean", outfile="fig2_fingerprint_no2.png")
