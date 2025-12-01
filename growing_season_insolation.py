#!/usr/bin/env python3
"""
Compute growing-season insolation on tree crowns using GRASS GIS.

- Uses a DSM as elevation input for r.sun
- Computes daily global radiation for a set of representative days
- Weights those days to approximate the whole growing season
- Sums them to a single raster of growing-season insolation
- Computes per-crown summary stats with v.rast.stats

Run this script INSIDE a GRASS session, e.g.:

    grass --text /path/to/LOCATION/MAPSET --exec python growing_season_insolation.py
"""

import sys
import math
import grass.script as gs

def main():
    # ------------------------------------------------------------------
    # USER PARAMETERS – CHANGE THESE TO MATCH YOUR DATA
    # ------------------------------------------------------------------
    dsm_raster = "ndsm_31370"              # name of your DSM raster in GRASS
    crowns_vector = "tree_layer_31370"   # name of your tree crown polygon layer in GRASS

    # Growing season definition (day of year)
    start_doy = 100                 # e.g. DOY 100 ~ mid April (depends on hemisphere)
    end_doy = 280                   # e.g. DOY 280 ~ early October
    day_step = 14                    # use 1 representative day every 7 days

    # Time-of-day settings for r.sun (in hours)
    start_time = 6.0                # from 6:00
    end_time = 18.0                 # to 18:00
    time_step = 2.0                 # hourly

    # Output names
    output_prefix = "insol"         # prefix for intermediate rasters
    final_raster = "insol_gs"       # final growing-season insolation raster
    column_prefix = "gsrad"         # prefix for stats columns in crown layer

    # If True, intermediate rasters are removed at the end
    cleanup_intermediate = True

    # ------------------------------------------------------------------
    # SCRIPT STARTS
    # ------------------------------------------------------------------
    gs.message("Starting growing-season insolation computation...")

    # Check that input maps exist
    if not gs.find_file(dsm_raster, element="cell")["name"]:
        gs.fatal(f"DSM raster <{dsm_raster}> not found in current mapset/LOCATION.")

    if not gs.find_file(crowns_vector, element="vector")["name"]:
        gs.fatal(f"Tree crowns vector <{crowns_vector}> not found in current mapset/LOCATION.")

    # Set computational region to DSM
    gs.message(f"Setting computational region to DSM <{dsm_raster}>")
    gs.run_command("g.region", raster=dsm_raster)

    # Generate list of representative DOYs
    representative_days = list(range(start_doy, end_doy + 1, day_step))
    if representative_days[-1] != end_doy:
        representative_days.append(end_doy)

    gs.message(f"Representative days: {representative_days}")

    # Compute how many days each representative DOY stands for
    # Simple approach: midpoints between DOYs
    weights = {}
    for i, doy in enumerate(representative_days):
        if len(representative_days) == 1:
            # Only one day: it represents the whole season
            weights[doy] = end_doy - start_doy + 1
        else:
            if i == 0:
                next_doy = representative_days[i + 1]
                midpoint = (doy + next_doy) / 2.0
                weight = int(round(midpoint - start_doy))
            elif i == len(representative_days) - 1:
                prev_doy = representative_days[i - 1]
                midpoint = (prev_doy + doy) / 2.0
                weight = int(round(end_doy - midpoint + 1))
            else:
                prev_doy = representative_days[i - 1]
                next_doy = representative_days[i + 1]
                mid_prev = (prev_doy + doy) / 2.0
                mid_next = (doy + next_doy) / 2.0
                weight = int(round(mid_next - mid_prev))
        # Enforce at least 1 day
        weights[doy] = max(1, weight)

    gs.message("Day-of-year weights (days represented by each DOY):")
    for doy in representative_days:
        gs.message(f"  DOY {doy}: {weights[doy]} days")

    # Sanity check: total days ~= end_doy - start_doy + 1
    total_weighted_days = sum(weights.values())
    season_days = end_doy - start_doy + 1
    gs.message(f"Total weighted days: {total_weighted_days} (target: {season_days})")

    # ------------------------------------------------------------------
    # 1) Compute daily global radiation for representative days with r.sun
    # ------------------------------------------------------------------
    daily_rasters = []

    for doy in representative_days:
        out_daily = f"{output_prefix}_doy_{doy}"
        gs.message(f"Computing daily global radiation for DOY {doy} -> <{out_daily}>")

        # r.sun: daily global radiation (Wh/m2) for one day of year
        # -s: use shadowing from terrain
        gs.run_command(
            "r.sun",
            elevation=dsm_raster,
            glob_rad=out_daily,
            day=doy,
            step=time_step,
            # You can add additional parameters here if needed, e.g. linke turbidity
            overwrite=True
        )

        daily_rasters.append(out_daily)

    # ------------------------------------------------------------------
    # 2) Weight each daily raster by the number of days it represents
    #    and sum to a growing-season raster
    # ------------------------------------------------------------------
    weighted_rasters = []

    for doy, rast in zip(representative_days, daily_rasters):
        weight = weights[doy]
        weighted_name = f"{rast}_w"
        gs.message(f"Weighting raster <{rast}> by {weight} days -> <{weighted_name}>")

        # Mapcalc: weighted_name = rast * weight
        gs.mapcalc(
            f"{weighted_name} = {rast} * {weight}",
            overwrite=True
        )
        weighted_rasters.append(weighted_name)

    # Sum all weighted rasters to final raster
    gs.message(f"Summing weighted rasters into final growing-season raster <{final_raster}>")

    gs.run_command(
        "r.series",
        input=",".join(weighted_rasters),
        output=final_raster,
        method="sum",
        overwrite=True
    )

    # ------------------------------------------------------------------
    # 3) Compute per-crown stats (mean, sum, etc.) with v.rast.stats
    # ------------------------------------------------------------------
    gs.message(f"Computing per-crown statistics from <{final_raster}> into <{crowns_vector}>")

    # -c: create new columns with prefix
    # You can change 'method' to include more stats (e.g. 'mean,sum,maximum')
    gs.run_command(
        "v.rast.stats",
        map=crowns_vector,
        raster=final_raster,
        column_prefix=column_prefix,
        method="average,sum",
        flags="c"
    )

    gs.message("Per-crown insolation statistics written to attribute table.")
    gs.message(f"Columns should look like: {column_prefix}_mean, {column_prefix}_sum")

    # ------------------------------------------------------------------
    # 4) Optional cleanup of intermediate rasters
    # ------------------------------------------------------------------
    if cleanup_intermediate:
        gs.message("Cleaning up intermediate rasters...")
        to_remove = daily_rasters + weighted_rasters
        gs.run_command("g.remove", type="raster", name=",".join(to_remove), flags="f")

    gs.message("Done! Growing-season insolation per tree crown is ready.")


if __name__ == "__main__":
    sys.exit(main())
