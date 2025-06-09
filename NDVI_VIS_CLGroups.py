import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime
import os
import warnings

# === INPUT / OUTPUT ===
input_path = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Sampelen/NDVI_sub/NDVI_subselectie.csv"
output_csv = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Sampelen/NDVI_sub/sos_NDVI_per_boom_sub.csv"
output_plot_dir = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Sampelen/NDVI_sub/output_plots_NDVI_sub"
os.makedirs(output_plot_dir, exist_ok=True)

# === DATA INLEZEN ===
df = pd.read_csv(input_path)
date_cols = df.columns[2:]
date_doys = [datetime.datetime.strptime(d, "%d_%m_%Y").timetuple().tm_yday for d in date_cols]
t_obs = np.array(date_doys) / 365

# === DOUBLE LOGISTIC + DERIVATIE ===
def double_logistic_function(t, wNDVI, mNDVI, S, A, mS, mA):
    sig1 = 1 / (1 + np.exp(np.clip(-mS * (t - S), -100, 100)))
    sig2 = 1 / (1 + np.exp(np.clip(mA * (t - A), -100, 100)))
    return wNDVI + (mNDVI - wNDVI) * (sig1 + sig2 - 1)

def forward_difference(x, y):
    der = [(y[i+1] - y[i]) / (x[i+1] - x[i]) for i in range(len(x)-1)]
    der.append(der[-1])
    return np.array(der)

def fit_curve(t, y, tree_id=None):
    try:
        wNDVI = np.clip(np.percentile(y, 10), 0.05, 0.95)
        mNDVI = np.clip(np.percentile(y, 90), 0.1, 0.99)
        S = np.clip(t[np.argmax(np.gradient(y))], 0.05, 0.95)
        A = np.clip(t[np.argmin(np.gradient(y))], 0.05, 0.95)
        mS = 10
        mA = 10
        guess = [wNDVI, mNDVI, S, A, mS, mA]

        bounds = (
            [0, 0, 0.01, 0.01, 0.5, 0.5],
            [1, 1, 0.99, 0.99, 100, 100]
        )

        params, _ = curve_fit(double_logistic_function, t, y, p0=guess, bounds=bounds, method='trf', max_nfev=100000)
        return params
    
    except Exception as e:
        if tree_id is not None:
            print(f"Fout bij fitten (Tree {tree_id}): {e}")
        else:
            print(f"Fout bij fitten: {e}")
        return None

def get_first_peak(array):
    for i in range(len(array)-1):
        if array[i+1] < array[i] and i > 5:
            return i
    return 0

# === SOS ANALYSE PER BOOM ===
results = []
for idx, row in df.iterrows():
    tree_id = row["field_1"]
    cluster_id = row["CL_ID"]
    ndvi = row[date_cols].values.astype(float)

    params = fit_curve(t_obs, ndvi, tree_id=tree_id)

    if params is not None:
        t_fit = np.linspace(t_obs.min(), t_obs.max(), 1000)
        ndvi_fit = double_logistic_function(t_fit, *params)
        deriv3 = forward_difference(t_fit, forward_difference(t_fit, forward_difference(t_fit, ndvi_fit)))
        peak_idx = get_first_peak(deriv3)
        sos_doy = int(t_fit[peak_idx] * 365)

        # === PLOT MAKEN ===
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(t_obs * 365, ndvi, label="Observed NDVI", color='blue')
        ax.plot(t_fit * 365, ndvi_fit, label="Fitted curve", color='green')
        ax.axvline(sos_doy, color='red', linestyle='--', label=f'SOS: DOY {sos_doy}')
        ax.set_title(f"Tree {tree_id} (Cluster {cluster_id})")
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("NDVI")
        ax.legend()
        plt.tight_layout()

        # === Cluster-submap aanmaken en plot opslaan ===
        cluster_dir = os.path.join(output_plot_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        fig.savefig(os.path.join(cluster_dir, f"tree_{tree_id}.png"))
        plt.close()
    else:
        sos_doy = None

    results.append({
        "field_1": tree_id,
        "CL_ID": cluster_id,
        "SOS": sos_doy
    })

# === RESULTAAT OPSLAAN ===
sos_df = pd.DataFrame(results)
sos_df.to_csv(output_csv, index=False)
print(f"SOS per boom opgeslagen in: {output_csv}")
print(f"Visualisaties opgeslagen in map: {output_plot_dir}")
