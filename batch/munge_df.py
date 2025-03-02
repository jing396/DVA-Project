# %%
import pandas as pd
import numpy as np


def munge_df(df):
    """
    Returns a dataframe after uppercasing column names and merging
    weather station and imputed zipcodes.

    Parameters
    ----------
    df : DataFrame
        The NYC crash dataset

    Returns
    -------
    DataFrame
        With new columns `LATITUDE_WEATHER_STATION`, `LONGITUDE_WEATHER_STATION`,
        `DISTANCE_FROM_WEATHER_STATION`, and `ZIP_CODE_IMPUTED`.
    """
    # Replace column name spaces with underscore
    df.columns = df.columns.str.replace(" ", "_")

    # Parse dates
    df["CRASH_DATE"] = pd.to_datetime(df["CRASH_DATE"], errors='coerce').dt.strftime("%Y-%m-%d")

    missing_prop = (
        df[df["ZIP_CODE"].isnull() & df["LOCATION"].isnull()].shape[0] / df.shape[0]
    ) * 100
    print(f"{missing_prop}% of the rows does not have any of these: zip code, lat/lon")

    df_nearest = pd.concat(
        [
            pd.read_csv("data/weather_station_mapping_part1.csv"),
            pd.read_csv("data/weather_station_mapping_part2.csv"),
        ]
    )

    df = df.merge(
        df_nearest,
        left_on=["CRASH_DATE", "LATITUDE", "LONGITUDE"],
        right_on=["DATE", "LATITUDE_ACCIDENT", "LONGITUDE_ACCIDENT"],
        how="left",
        validate="m:1",
    )
    weather_station_prop = (
        df[~df["STATION"].isnull()].shape[0] / df[~df["LOCATION"].isnull()].shape[0]
    ) * 100

    print(
        f"Weather station names added to {weather_station_prop}% of crashes containing lat/lon information"
    )

    df_null_zip = pd.read_csv("data/imputed_zip.csv", dtype={"ZIP_CODE_IMPUTED": "str"})

    # Link missing zipcodes (only if lat/lon is present)
    df = df.merge(df_null_zip, on=["LATITUDE", "LONGITUDE"], how="left", validate="m:1")

    # Impute zip codes
    print(
        f"{round((df[df['ZIP_CODE'].isnull()].shape[0]/df.shape[0])*100,2)}% of the data has missing zip codes prior to imputing"
    )
    df["ZIP_CODE"] = np.where(
        df["ZIP_CODE"].isnull(), df["ZIP_CODE_IMPUTED"], df["ZIP_CODE"]
    )
    print(
        f"{round((df[df['ZIP_CODE'].isnull()].shape[0]/df.shape[0])*100,2)}% of the data has missing zip codes after imputing"
    )
    locations_prop = df[~df["LOCATION"].isnull()]
    print(
        f"{round((locations_prop[~locations_prop['ZIP_CODE'].isnull()].shape[0]/locations_prop.shape[0])*100,2)}% of the rows with lat/lon contain zip codes after imputing"
    )

    # Clean up zip codes
    df["ZIP_CODE"] = df["ZIP_CODE"].str.replace(" ", "")

    return df
