import glob
import math

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

###########################
# IMPORT DATA
###########################
# Get NYC df
df = pd.read_csv(
    "data/Motor_Vehicle_Collisions_-_Crashes.csv", dtype={"ZIP CODE": "str"}
)
# Replace column name spaces with underscore
df.columns = df.columns.str.replace(" ", "_")

# Parse dates
df["CRASH_DATE"] = pd.to_datetime(df.CRASH_DATE).dt.strftime("%Y-%m-%d")

# Filter null coordinates
df = df[~df["LOCATION"].isnull()]

# Get weather station df
files = glob.glob("data/NY Weather*.csv")
df_weather = pd.concat([pd.read_csv(file) for file in files])

# Filter crash data with only available weather data
df = df[df["CRASH_DATE"].isin(set(df_weather.DATE.unique()))]

# # Remove duplicates
df = df[~df[["CRASH_DATE", "LATITUDE", "LONGITUDE"]].duplicated()].reset_index(
    drop=True
)

# Limit columns
df = df[["CRASH_DATE", "LATITUDE", "LONGITUDE"]]
df_weather = df_weather[["DATE", "STATION", "LATITUDE", "LONGITUDE"]]

###########################
# CREATE MAPPING CSV
###########################
# %%
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["LATITUDE"], df["LONGITUDE"]),
)
gdf_weather = gpd.GeoDataFrame(
    df_weather,
    geometry=gpd.points_from_xy(df_weather["LATITUDE"], df_weather["LONGITUDE"]),
)

cuts = {k: d for k, d in gdf_weather.groupby("DATE")}
# %%
df_nearest = (
    gdf.groupby("CRASH_DATE")
    .progress_apply(
        lambda d: gpd.sjoin_nearest(
            d,
            cuts[d["CRASH_DATE"].values[0]],
            distance_col="DISTANCE_FROM_WEATHER_STATION",
            lsuffix="ACCIDENT",
            rsuffix="WEATHER_STATION",
        )
    )
    .reset_index(drop=True)
)
df_nearest = df_nearest.drop(
    columns=["geometry", "index_WEATHER_STATION", "CRASH_DATE"]
)


def save_df_in_chunks(df, base_filename, rows_per_chunk):
    num_chunks = math.ceil(len(df) / rows_per_chunk)

    for i in range(num_chunks):
        start_index = i * rows_per_chunk
        end_index = min((i + 1) * rows_per_chunk, len(df))
        df_chunk = df[start_index:end_index]
        chunk_filename = f"{base_filename}_part{i+1}.csv"
        df_chunk.to_csv(chunk_filename, index=False)


save_df_in_chunks(
    df_nearest, "data/weather_station_mapping", int(df_nearest.shape[0] / 2)
)
# %%
