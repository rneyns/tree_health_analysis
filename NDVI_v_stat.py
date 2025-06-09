import pandas as pd
import numpy as np

# === 1. DATA INLEZEN ===
bestand = "/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Stat/NDVI_PM25.csv"
df = pd.read_csv(bestand)

# === 2. LONG FORMAT VAN NDVI-DATA ===
ndvi_cols = [col for col in df.columns if col.startswith('NDVI_subselectie_') and col[-4:].isdigit()]

df_long = df.melt(id_vars=['field_1', 'CL_ID', 'SAMPLE_1_2'],
                  value_vars=ndvi_cols,
                  var_name='Date', value_name='NDVI')

df_long['Date'] = df_long['Date'].str.replace('NDVI_subselectie_', '', regex=False)
df_long['Date'] = pd.to_datetime(df_long['Date'], format='%d_%m_%Y')
df_long = df_long.sort_values(['field_1', 'Date'])

# === 3. PIEKWAARDEN PER BOOM ===
def bepaal_piek(g):
    max_idx = g['NDVI'].idxmax()
    return pd.Series({
        'NDVI_max': g.loc[max_idx, 'NDVI'],
        'NDVI_max_date': g.loc[max_idx, 'Date']
    })

pieken = (
    df_long
    .groupby('field_1')
    .apply(lambda g: bepaal_piek(g.drop(columns='field_1')))
    .reset_index()
)

# === 4. AFNAMEGRAAD NA PIEK ===
def afnamegraad_na_piek(g):
    g = g.dropna(subset=['NDVI'])
    if g.empty:
        return np.nan
    max_idx = g['NDVI'].idxmax()
    peak_date = g.loc[max_idx, 'Date']
    na_piek = g[g['Date'] > peak_date]
    if len(na_piek) < 2:
        return np.nan
    dagen = (na_piek['Date'] - peak_date).dt.days
    return np.polyfit(dagen, na_piek['NDVI'], 1)[0]

afnames = (
    df_long
    .groupby('field_1')
    .apply(lambda g: pd.Series({'NDVI_afnamegraad': afnamegraad_na_piek(g.drop(columns='field_1'))}))
    .reset_index()
)

# === 5. SAMENVOEGEN EN FORMATTEREN ===
metadata = df[['field_1', 'CL_ID', 'SAMPLE_1_2']]
resultaat = metadata.merge(pieken, on='field_1').merge(afnames, on='field_1')

# Datumkolom proper maken
resultaat['NDVI_max_date'] = resultaat['NDVI_max_date'].dt.date

# OPSLAAN ALS EXCEL 
resultaat.to_excel("/Users/alexsamyn/Documents/BAP_(Mac)/BAP_New/Stat/NDVI_voorbereiding/NDVI_v_stat.xlsx", index=False)

print(" Analyse voltooid en opgeslagen als 'NDVI_v_stat.xlsx'")

