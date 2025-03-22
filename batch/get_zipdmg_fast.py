# %%
from census import Census
import pandas as pd
import numpy as np

CENSUS_API_KEY = (
    # Replace with your actual API key
)
c = Census(CENSUS_API_KEY)
data = c.acs5.get(
    ("NAME", "GEO_ID", "B01003_001E", "B01003_001M", "B25034_010E", "B25034_010M"),
    {"for": "zip code tabulation area:*"},
)
# %%
df = pd.DataFrame(data)

df["zipcode"] = df["GEO_ID"].apply(lambda x: x.split("US")[-1])
df["land_area_sq_mi"] = df["B25034_010E"].astype(float) * 0.0000003861

df = df.rename(
    columns={
        "B01003_001E": "population",
        "B01003_001M": "population_error",
        "B25034_010E": "land_area_sq_meter",
        "B25034_010M": "land_area_error",
        "zip code tabulation area": "ZCTA",
        "zipcode": "ZIP_CODE",
    }
)

df["pop_density_meters"] = df["population"] / df["land_area_sq_meter"]
df["pop_density_meters"] = np.where(
    df["pop_density_meters"] == np.inf, 0, df["pop_density_meters"]
)


df = df[
    [
        "ZIP_CODE",
        "pop_density_meters",
        "population",
        "population_error",
        "land_area_sq_meter",
        "land_area_error",
        "ZCTA",
    ]
]

df.to_csv("data/population_lookup.csv", index=False)
