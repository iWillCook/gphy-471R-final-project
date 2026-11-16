"""
02c_daymet.py
Build the Daymet climate predictors at each SNOTEL station: fill the
leap-year gap, then compute cumulative water-year precipitation, 7-day
precipitation, and accumulated positive degree-days.
Output: data/processed/daymet_features.csv
"""

import pandas as pd
import config as c

c.make_dirs()


def fill_leap_days(df):
    """Add any missing days (Daymet drops Dec 31 in leap years).

    df: table with id, date, prcp, tmax, tmin
    Returns the table with one row per station per day: temperatures
    interpolated from the neighboring days, precipitation set to 0.
    """
    start, end = c.date_span()
    days = pd.date_range(start, end, freq="D")
    full = pd.MultiIndex.from_product([df["id"].unique(), days], names=["id", "date"])
    df = df.set_index(["id", "date"]).reindex(full).reset_index()

    df["prcp"] = df["prcp"].fillna(0)
    for col in ["tmax", "tmin"]:
        df[col] = df.groupby("id")[col].transform(lambda s: s.interpolate())
    return df


def add_features(df):
    """Add the climate predictors to a daily table.

    df: table with id, date, prcp (mm/day), tmax and tmin (deg C), one row per day
    Returns the table with wy, cum_prcp, prcp_7d, and pdd added.
    Works for station tables now and for the gridded Cottonwood data later.
    """
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    df["wy"] = df["date"].dt.year + (df["date"].dt.month >= 10)   # Oct-Dec -> next water year

    # Cumulative precipitation since Oct 1 (mm)
    df["cum_prcp"] = df.groupby(["id", "wy"])["prcp"].cumsum()

    # Precipitation over the last 7 days, including this one (mm)
    df["prcp_7d"] = (df.groupby("id")["prcp"]
                       .transform(lambda s: s.rolling(7, min_periods=7).sum()))

    # Accumulated positive degree-days since Oct 1 (deg C * days)
    # Accumulated positive degree-days since Dec 1 (deg C * days). Counting from
    # Oct 1 would be dominated by warm fall days before the snowpack forms.
    tmean = (df["tmax"] + df["tmin"]) / 2
    warm = tmean.clip(lower=0).where(df["date"].dt.month.isin([12, 1, 2, 3]), 0)
    df["pdd"] = warm.groupby([df["id"], df["wy"]]).cumsum()
    return df


# ---------------- Run ----------------
daymet = pd.read_csv(c.daymet_dir / "daymet_stations.csv", parse_dates=["date"])
n_before = len(daymet)

elev = daymet.groupby("id")["elev_m"].first()   # Daymet cell elevation, kept for reference
daymet = fill_leap_days(daymet[["id", "date", "prcp", "tmax", "tmin"]])
print(f"Filled {len(daymet) - n_before} missing days (leap-year gap)")

daymet = add_features(daymet)
daymet["daymet_elev_m"] = daymet["id"].map(elev)

out = c.processed / "daymet_features.csv"
daymet.round(2).to_csv(out, index=False)
print(f"Saved {len(daymet):,} station-days to {out.name}")
print(daymet[["cum_prcp", "prcp_7d", "pdd"]].describe().round(1))