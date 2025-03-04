# %%
import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from tqdm.auto import tqdm

tqdm.pandas()
# %%
###########################
# IMPORT DATA
###########################
df = pd.read_csv(
    "data/Motor_Vehicle_Collisions_-_Crashes.csv", dtype={"ZIP CODE": "str"}
)
# df = df.head(20)
# Replace column name spaces with underscore
df.columns = df.columns.str.replace(" ", "_")

df = df[["LATITUDE", "LONGITUDE", "LOCATION", "ZIP_CODE"]]

# Filter data with known lat/lon but with null zip codes
df_null_zip = df[
    (~df[["LOCATION"]].isnull().all(axis=1)) & (df["ZIP_CODE"].isnull())
].copy()

# Drop duplicates
df_null_zip = df_null_zip[~df_null_zip[["LOCATION"]].duplicated()]

print(f"There are {df_null_zip.shape[0]} unique lat/lon having null zip codes")

# %%
############################
# REVERSE ZIP CODE LOOKUP
############################
geolocator = Nominatim(user_agent="reverse_geocoding")
reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

# Keep lookup in memory to reduce API calls
imputed = pd.read_csv("data/cache/imputed_zip.csv", dtype={"ZIP_CODE_IMPUTED": "str"})
reverse_dict = dict(zip(imputed["LOCATION"], imputed["ZIP_CODE_IMPUTED"]))


# Helper function
def impute(coord):
    if not reverse_dict.get(coord):
        geo_result = reverse_geocode(coord.strip("()"))
        if geo_result:
            zip = geo_result.raw["address"].get("postcode")
            reverse_dict[coord] = str(zip)
            pd.DataFrame(
                reverse_dict.items(), columns=["LOCATION", "ZIP_CODE_IMPUTED"]
            ).to_csv("data/cache/imputed_zip.csv", index=False)
            return zip
        else:
            return None
    else:
        # print(f"{coord}:{reverse_dict[coord]} exists")
        return reverse_dict.get(coord)


df_null_zip["ZIP_CODE_IMPUTED"] = df_null_zip["LOCATION"].progress_apply(impute)
# %%
# Save lookup to csv
df_null_zip[["LATITUDE", "LONGITUDE", "ZIP_CODE_IMPUTED"]].to_csv(
    "data/imputed_zip.csv", index=False
)

# %%
