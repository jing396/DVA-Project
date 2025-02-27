# %%
from batch.munge_df import munge_df

import pandas as pd

import glob

# Import data
df = pd.read_csv(
    "data/Motor_Vehicle_Collisions_-_Crashes.csv", dtype={"ZIP CODE": "str"}
)

# Munge data
df = munge_df(df)

# Limit rows for demo
df = df[~df["LOCATION"].isnull()]

# Get weather station df
files = glob.glob("data/NY Weather*.csv")
df_weather = pd.concat([pd.read_csv(file) for file in files])

# Merge with weather data using crash date and nearest weather station
df = df.merge(
    df_weather,
    left_on=["CRASH_DATE", "STATION"],
    right_on=["DATE", "STATION"],
    how="inner",
    validate="m:1",
    suffixes=["_CRASH", "_WEATHER_STATION"],
)
# %%
df
