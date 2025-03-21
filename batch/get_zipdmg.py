# %%
from batch.munge_df import munge_df

from census import Census
import pandas as pd
import time
from tqdm import tqdm

import os

# %%
os.chdir("D:/dva-project/DVA-Project/")

# Import data
df = pd.read_csv(
    "data/Motor_Vehicle_Collisions_-_Crashes.csv", dtype={"ZIP CODE": "str"}
)

# Munge data
df = munge_df(df)

# %%
df["CRASH_YEAR"] = df.CRASH_DATE.str.slice(0, 4).astype(int)
df_zipcodes = df[~df["ZIP_CODE"].isnull()]
df_zipcodes = df_zipcodes[["ZIP_CODE", "CRASH_YEAR"]]
df_zipcodes = df_zipcodes[~df_zipcodes.duplicated()]
df_zipcodes
# %%

# Replace with your actual API key
CENSUS_API_KEY = "YOUR_API_KEY"

c = Census(CENSUS_API_KEY)


def get_population_data(zip_tuples, retries=3, delay=10):
    """Retrieves population data for a list of zip codes, handling rate limits."""
    all_data = []
    for zip_code, year in tqdm(zip_tuples):
        for attempt in range(retries):
            try:
                data = c.acs5.get(
                    ("NAME", "B01003_001E"),
                    {"for": f"zip code tabulation area:{zip_code}"},
                    year=year,
                )
                if data:
                    all_data.extend(data)
                else:
                    print(f"No data found for zip code {zip_code}")
                break  # If successful, exit retry loop
            except Exception as e:
                print(f"Error for zip code {zip_code} (Attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)  # Wait before retrying
                else:
                    print(
                        f"Failed to retrieve data for {zip_code} after {retries} attempts."
                    )
    return all_data


population_data = get_population_data(df_zipcodes.apply(tuple, axis=1))

if population_data:
    df_pop = pd.DataFrame(population_data)
    df_pop = df.rename(
        columns={"B01003_001E": "Population", "zip code tabulation area": "Zipcode"}
    )
    df_pop.to_csv("data/population_lookup.csv")
    print(df[["Zipcode", "Population"]])
else:
    print("No population data retrieved.")

# %%
