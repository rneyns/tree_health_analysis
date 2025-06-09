import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime
import os
import matplotlib.cm as cm
import warnings

# === PADEN EN MAPPEN ===

input_path = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Sampelen/MTVI2/MTVI2_per_boom.csv"
output_csv = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Sampelen/MTVI2/sos_NDVI_per_boom_sub.csv"
output_plot_dir = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Sampelen/MTVI2/output_plots_MTVI2"
output_cluster_dir = os.path.join(os.path.dirname(output_plot_dir), "output_plots_clusters_MTVI2")

os.makedirs(output_plot_dir, exist_ok=True)
os.makedirs(output_cluster_dir, exist_ok=True)

# === DATA INLEZEN ===
df = pd.read_csv(input_path)
date_cols = df.columns[2:]
date_doys = [datetime.datetime.strptime(d, "%d_%m_%Y").timetuple().tm_yday for d in date_cols]
t_obs = np.array(date_doys) / 365

# === FITTING EN SOS-FUNCTIES ===
def double_logistic_function(t, wNDVI, mNDVI, S, A, mS, mA):
    sig1 = 1 / (1 + np.exp(np.clip(-mS * (t - S), -100, 100)))
    sig2 = 1 / (1 + np.exp(np.clip(mA * (t - A), -100, 100)))
    return wNDVI + (mNDVI - wNDVI) * (sig1 + sig2 - 1)

def forward_difference(x, y):
    der = [(y[i+1] - y[i]) / (x[i+1] - x[i]) for i in range(len(x)-1)]
    der.append(der[-1])
    return np.array(der)

def fit_curve(t, y):
    try:
        wNDVI = np.clip(np.percentile(y, 10), 0.05, 0.95)
        mNDVI = np.clip(np.percentile(y, 90), 0.1, 0.99)
        S = np.clip(t[np.argmax(np.gradient(y))], 0.05, 0.95)
        A = np.clip(t[np.argmin(np.gradient(y))], 0.05, 0.95)
        mS = 10
        mA = 10
        guess = [wNDVI, mNDVI, S, A, mS, mA]
        bounds = ([0, 0, 0.01, 0.01, 0.5, 0.5], [1, 1, 0.99, 0.99, 100, 100])
        params, _ = curve_fit(double_logistic_function, t, y, p0=guess, bounds=bounds, method='trf', max_nfev=100000)
        return params
    except Exception as e:
        print(f"Fout bij fitten: {e}")
        return None

def get_first_peak(array):
    for i in range(len(array)-1):
        if array[i+1] < array[i] and i > 5:
            return i
    return 0

# === SOS-BEREKENING PER BOOM + PLOT ===
results = []
for idx, row in df.iterrows():
    tree_id = row["field_1"]
    cluster_id = row["CL_ID"]
    ndvi = row[date_cols].values.astype(float)

    if np.std(ndvi) < 0.05 or np.all(np.isnan(ndvi)):
        sos_doy = None
        results.append({"field_1": tree_id, "CL_ID": cluster_id, "SOS": sos_doy})
        continue

    params = fit_curve(t_obs, ndvi)

    if params is not None:
        t_fit = np.linspace(t_obs.min(), t_obs.max(), 1000)
        ndvi_fit = double_logistic_function(t_fit, *params)
        deriv3 = forward_difference(t_fit, forward_difference(t_fit, forward_difference(t_fit, ndvi_fit)))
        sos_idx = get_first_peak(deriv3)
        sos_doy = int(t_fit[sos_idx] * 365)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(t_obs * 365, ndvi, label="Observed MTVI2", color='blue')
        ax.plot(t_fit * 365, ndvi_fit, label="Fitted curve", color='green')
        ax.axvline(sos_doy, color='red', linestyle='--', label=f'SOS: DOY {sos_doy}')
        ax.set_title(f"Tree {tree_id} (Cluster {cluster_id})")
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("MTVI2")
        ax.legend()
        plt.tight_layout()
        fig.savefig(f"{output_plot_dir}/tree_{tree_id}_cluster_{cluster_id}.png")
        plt.close()
    else:
        sos_doy = None

    results.append({"field_1": tree_id, "CL_ID": cluster_id, "SOS": sos_doy})

sos_df = pd.DataFrame(results)
sos_df.to_csv(output_csv, index=False)
print(f"SOS per boom opgeslagen in: {output_csv}")
print(f"Figuur per boom opgeslagen in map: {output_plot_dir}")

# === CLUSTERVISUALISATIES ===
df_merged = pd.merge(df, sos_df, on=["field_1", "CL_ID"])
t_fit = np.linspace(t_obs.min(), t_obs.max(), 1000)

for cluster_id, group in df_merged.groupby("CL_ID"):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = cm.viridis(np.linspace(0, 1, len(group)))

    for i, (_, row) in enumerate(group.iterrows()):
        tree_id = row["field_1"]
        ndvi_vals = row[date_cols].values.astype(float)
        params = fit_curve(t_obs, ndvi_vals)
        sos_doy = row["SOS"]

        if params is not None:
            curve = double_logistic_function(t_fit, *params)
            ax.plot(t_fit * 365, curve, color=colors[i], linewidth=1, alpha=0.8)
            if not pd.isna(sos_doy):
                ax.axvline(sos_doy, linestyle='--', color=colors[i], linewidth=0.8)

    ax.set_title(f"MTVI2 Season Curves – Cluster {cluster_id}")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("MTVI2")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_cluster_dir}/cluster_{cluster_id}_overview.png")
    plt.close()

print("Clusterplots opgeslagen in map:", output_cluster_dir)
