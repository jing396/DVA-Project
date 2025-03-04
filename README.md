# DVA-Project
Team 169
NY weather csv data source: climate.gov https://www.climate.gov/maps-data/dataset/past-weather-zip-code-data-table
NY weather data description: GHCND_documentation.pdf

# Building and using environment
To create environment, run `conda env create -f builds/environment.yml` at the root directory and then activating it via `conda activate dva`.

Example:
```bash
conda env create -f builds/environment.yml
conda activate dva
```

If you install a new package, run `conda env export -f builds/environment.yml`
and then commit this file and push to github (either through the terminal or through an editor) so that others can utilize the same package and version.

Example:
```bash
git checkout my_own_branch
conda env export -f builds/environment.yml
git add builds/environment.yml
git commit -m "update conda env"
git push
```

# Data Munging 
You can run either code block to get weather station and missing zip codes:

## Method 1: Using `munge_df()`
```python
from batch.munge_df import munge_df

import pandas as pd

df = pd.read_csv(
    "data/Motor_Vehicle_Collisions_-_Crashes.csv", dtype={"ZIP CODE": "str"}
)

df = munge_df(df)
```

## Method 2: Manually

```python
import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/Motor_Vehicle_Collisions_-_Crashes.csv", dtype={"ZIP CODE": "str"}
)
# Replace column name spaces with underscore
df.columns = df.columns.str.replace(" ", "_")

# Link weather station
df_nearest = pd.read_csv("data/weather_station_mapping.csv")

df = df.merge(
    df_nearest,
    left_on=["LATITUDE", "LONGITUDE"],
    right_on=["LATITUDE_ACCIDENT", "LONGITUDE_ACCIDENT"],
    how="left",
    validate="m:1",
)

df_null_zip = pd.read_csv("data/imputed_zip.csv", dtype={"ZIP_CODE_IMPUTED": "str"})

# Link missing zipcodes (only if lat/lon is present)
df = df.merge(df_null_zip, on=["LATITUDE", "LONGITUDE"], how="left", validate="m:1")

df["ZIP_CODE"] = np.where(
    df["ZIP_CODE"].isnull(), df["ZIP_CODE_IMPUTED"], df["ZIP_CODE"]
)
```