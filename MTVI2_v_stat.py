import pandas as pd
import numpy as np

# === 1. DATA INLEZEN ===
bestand = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Stat/MTVI2_PM25.csv"
df = pd.read_csv(bestand)

# === 2. LONG FORMAT VAN MTVI2-DATA ===
mtvi2_cols = [col for col in df.columns if col.startswith('MTVI2_') and col[-4:].isdigit()]

df_long = df.melt(id_vars=['field_1', 'CL_ID', 'SAMPLE_1_2'],
                  value_vars=mtvi2_cols,
                  var_name='Date', value_name='MTVI2')

df_long['Date'] = df_long['Date'].str.replace('MTVI2_per_boom_', '', regex=False)
df_long['Date'] = pd.to_datetime(df_long['Date'], format='%d_%m_%Y')
df_long = df_long.sort_values(['field_1', 'Date'])

# === 3. PIEKWAARDEN PER BOOM ===
def bepaal_piek(g):
    max_idx = g['MTVI2'].idxmax()
    return pd.Series({
        'MTVI2_max': g.loc[max_idx, 'MTVI2'],
        'MTVI2_max_date': g.loc[max_idx, 'Date']
    })

pieken = (
    df_long
    .groupby('field_1')
    .apply(bepaal_piek)
    .reset_index()
)

# === 4. AFNAMEGRAAD NA PIEK ===
def afnamegraad_na_piek(g):
    g = g.dropna(subset=['MTVI2'])
    if g.empty:
        return np.nan
    max_idx = g['MTVI2'].idxmax()
    peak_date = g.loc[max_idx, 'Date']
    na_piek = g[g['Date'] > peak_date]
    if len(na_piek) < 2:
        return np.nan
    dagen = (na_piek['Date'] - peak_date).dt.days
    return np.polyfit(dagen, na_piek['MTVI2'], 1)[0]

afnames = (
    df_long
    .groupby('field_1')
    .apply(lambda g: pd.Series({'MTVI2_afnamegraad': afnamegraad_na_piek(g)}))
    .reset_index()
)

# === 5. SAMENVOEGEN EN FORMATTEREN ===
metadata = df[['field_1', 'CL_ID', 'SAMPLE_1_2']]
resultaat = metadata.merge(pieken, on='field_1').merge(afnames, on='field_1')

# Datumkolom proper maken
resultaat['MTVI2_max_date'] = resultaat['MTVI2_max_date'].dt.date


output_path = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Stat/MTVI2_voorbereiding/MTVI2_v_stat.xlsx"
resultaat.to_excel(output_path, index=False)

print(" Analyse voltooid en opgeslagen als 'MTVI2_v_stat.xlsx'")
