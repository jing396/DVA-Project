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
df_weather = pd.concat([pd.read_csv(file, low_memory=False) for file in files])

# Merge with weather data using crash date and nearest weather station
df = df.merge(
    df_weather,
    left_on=["CRASH_DATE", "STATION"],
    right_on=["DATE", "STATION"],
    how="inner",
    validate="m:1",
    suffixes=["_CRASH", "_WEATHER_STATION"],
)
df.to_csv("data/df_demo.csv", index=False)
# %%
df

# print(df.head())

# print(df.shape)

df_prcp = df[df["PRCP"].notnull()].head(10)

df_snow = df[df["SNOW"].notnull()].head(10)

sample_20_rows = pd.concat([df_prcp, df_snow], ignore_index=True)

sample_20_rows.to_csv("sample_20_rows.csv", index=False)
