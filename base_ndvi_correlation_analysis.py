import geopandas as gpd, pandas as pd, numpy as np
from rasterstats import zonal_stats

# 1. Load tree layer and CSV with baseline NDVI
g = gpd.read_file("trees.shp")
csv = pd.read_csv("baseline_ndvi.csv")         # columns: id, baseline_ndvi
g = g.merge(csv, on="id")                      # join baseline values by ID

# 2. Compute neighbourhood vegetation fraction (3 m buffer)
g["geom_buf"] = g.geometry.buffer(3)
zs = zonal_stats(g.set_geometry("geom_buf"), "veg_mask.tif", stats=["mean"], nodata=0)
g["veg_frac"] = [z["mean"] for z in zs]

# 3. Correlation and quick sanity check
print("r =", np.corrcoef(g["baseline_ndvi"], g["veg_frac"])[0,1])

import statsmodels.api as sm
X = sm.add_constant(g[["veg_frac"]])  # or imperviousness
y = g["baseline_ndvi"]
print(sm.OLS(y, X).fit().summary())
