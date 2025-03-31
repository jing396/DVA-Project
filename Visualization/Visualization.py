import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import json
import geopandas as gpd
import branca

print("Current Working Directory:", os.getcwd())

df = pd.read_csv("cleaned_data_updated.csv", dtype={"ZIP CODE": "str"})
df["DAY_OF_WEEK"] = pd.to_datetime(df["CRASH_DATETIME"]).dt.day_name()
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
# print(df.head())

# ####################################################################################################################################################
# 1. Time trends analysis to identify accident trends over time

# 1.1. Create a figure with 4 subplots (year, months, day of the week, hour)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))


# Yearly Accident Trends (Line Chart)
# COVID Period start from March 2020 to May 2023
yearly_accidents = df[df["YEAR"] < 2025]["YEAR"].value_counts().sort_index()
sns.lineplot(x=yearly_accidents.index, y=yearly_accidents.values, marker="o", ax=axes[0, 0])
axes[0, 0].set_title("Yearly Accident Trends")
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("Number of Accidents")
axes[0, 0].axvspan(2020.25, 2023.417, color="gray", alpha=0.3)
axes[0, 0].text(2020.5, max(yearly_accidents.values) * 0.5, "COVID Period", fontsize=12, color="black", weight="bold")


# Monthly Accident Trends (Line Chart)
monthly_accidents = df["MONTH"].value_counts().sort_index()
top_3_months = monthly_accidents.nlargest(3)
sns.lineplot(x=monthly_accidents.index, y=monthly_accidents.values, marker="o", ax=axes[0, 1])

for month in top_3_months.index:
    axes[0, 1].scatter(month, monthly_accidents[month], color="red", s=100)

axes[0, 1].set_title("Monthly Accident Trends")
axes[0, 1].set_xlabel("Month")
axes[0, 1].set_ylabel("Number of Accidents")
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])


# Day of the Week Accident Trends (Line Chart)
day_of_week_accidents = df["DAY_OF_WEEK"].value_counts()[
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
]
top_day = day_of_week_accidents.idxmax()
top_day_value = day_of_week_accidents.max()
sns.lineplot(x=day_of_week_accidents.index, y=day_of_week_accidents.values, marker="o", ax=axes[1, 0])
axes[1, 0].scatter(top_day, top_day_value, color="red", s=100)
axes[1, 0].set_title("Accidents by Day of the Week")
axes[1, 0].set_xlabel("Day of the Week")
axes[1, 0].set_ylabel("Number of Accidents")


# Hourly Accident Trends (Line Chart)
hourly_accidents = df["HOUR"].value_counts().sort_index()
sns.lineplot(x=hourly_accidents.index, y=hourly_accidents.values, marker="o", ax=axes[1, 1])

rush_hours = [8, 16, 17]
for hour in rush_hours:
    axes[1, 1].scatter(hour, hourly_accidents[hour], color="red", s=100)

axes[1, 1].set_title("Accident Distribution by Hour")
axes[1, 1].set_xlabel("Hour of the Day")
axes[1, 1].set_ylabel("Number of Accidents")
axes[1, 1].set_xticks(range(0, 24))
axes[1, 1].axvspan(6, 10, color="gray", alpha=0.3)
axes[1, 1].text(5.5, max(hourly_accidents.values) * 0.2, "Morning Rush Hour", fontsize=12, color="black", weight="bold")
axes[1, 1].axvspan(15, 19, color="gray", alpha=0.3)
axes[1, 1].text(14.5, max(hourly_accidents.values) * 0.3, "Evening Rush Hour", fontsize=12, color="black", weight="bold")

plt.tight_layout()
plt.savefig("output/Time Trends.png", dpi=300, bbox_inches="tight")

# ####################################################################################################################################################
# # 2. Spatial analysis to identify accident hotspots across NYC

# ############### 2.1. Accident heatmap with zip code boundaries #################
# Define the path for NYC ZIP Code GeoJSON file downloaded from the following link
# https://www.kaggle.com/datasets/saidakbarp/nyc-zipcode-geodata?resource=download)
nyc_zip_geojson_path = "nyc-zip-code-tabulation-areas-polygons.geojson"

# Load the NYC ZIP Code GeoJSON file
with open(nyc_zip_geojson_path, "r") as f:
    nyc_zip_geojson = json.load(f)

# Standardize ZIP code format
df["ZIP_CODE"] = df["ZIP_CODE"].astype(str).str.split('.').str[0].str.zfill(5)

# Compute crash counts per ZIP code
zip_crash_counts = df["ZIP_CODE"].value_counts().to_dict()
# print(zip_crash_counts)

# Add crash counts to GeoJSON features
for feature in nyc_zip_geojson["features"]:
    zip_code = feature["properties"].get("postalCode", "").zfill(5)
    feature["properties"]["crash_count"] = zip_crash_counts.get(zip_code, 0)

# Initialize map centered around NYC with a light grey background
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="cartodb positron")

# Add ZIP code boundaries from the GeoJSON file
folium.GeoJson(
    nyc_zip_geojson,
    name="ZIP Code Boundaries",
    style_function=lambda feature: {
        "fillColor": "lightgrey",
        "color": "black",
        "weight": 0.5,
        "opacity": 0.5,
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=["postalCode", "crash_count"],
        aliases=["ZIP Code:", "Crashes:"],
        sticky=True,
        delay=500
    )
).add_to(nyc_map)

# Prepare heatmap data (drop NaN values)
heatmap_data = df[["LATITUDE_CRASH", "LONGITUDE_CRASH"]].dropna().values.tolist()

# Add Heatmap Layer
HeatMap(heatmap_data, radius=10, blur=10, min_opacity=0.2).add_to(nyc_map)

# Extract ZIP Code centroids and add text labels inside each ZIP code area
for feature in nyc_zip_geojson["features"]:
    if "geometry" in feature and "properties" in feature:
        zip_code = feature["properties"].get("postalCode", "Unknown")
        geometry = feature["geometry"]

        # Calculate centroid for labeling (for Polygon or MultiPolygon geometries)
        if geometry["type"] == "Polygon":
            coords = geometry["coordinates"][0]  # First ring of the polygon
        elif geometry["type"] == "MultiPolygon":
            coords = geometry["coordinates"][0][0]  # First ring of the first polygon
        else:
            continue

        # Compute centroid manually
        lon = sum(c[0] for c in coords) / len(coords)
        lat = sum(c[1] for c in coords) / len(coords)

        # Add ZIP Code label inside each ZIP code area
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html=f'<div style="font-size: 7pt; color: darkblue;">{zip_code}</div>'),
        ).add_to(nyc_map)

# Add Layer Control
folium.LayerControl().add_to(nyc_map)

# Save the interactive map
map_filename = "output/NYC_Accident_Heatmap_with_ZIP.html"
nyc_map.save(map_filename)

print(f"NYC Accident Heatmap with ZIP Code Boundaries: {map_filename}")


################ 2.2 Accident heatmap with borough boundaries #################
# Define the path for NYC Borough Boundaries GeoJSON file downloaded from the following link
# https://github.com/codeforgermany/click_that_hood/blob/main/public/data/new-york-city-boroughs.geojson
borough_geojson_path = "new-york-city-boroughs.geojson"

# Load the NYC Borough Boundaries GeoJSON as a GeoDataFrame
gdf_boroughs = gpd.read_file(borough_geojson_path)

# Load the NYC Borough Boundaries GeoJSON file manually
with open(borough_geojson_path, "r") as f:
    borough_geojson = json.load(f)

# Standardize borough names in dataset
df["BOROUGH"] = df["BOROUGH"].str.title()

# Compute crash counts per borough
borough_crash_counts = df["BOROUGH"].value_counts().to_dict()
# print(borough_crash_counts)

# Add crash count to the GeoJSON features
for feature in borough_geojson["features"]:
    borough_name = feature["properties"].get("name", "Unknown")
    feature["properties"]["crash_count"] = borough_crash_counts.get(borough_name, 0)

# Initialize map centered around NYC with a light grey background
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles='cartodb positron')

# Add borough boundaries with tooltips displaying crash counts
borough_layer = folium.GeoJson(
    borough_geojson,
    name="Borough Boundaries",
    style_function=lambda feature: {
        "fillColor": "lightgrey",
        "color": "black",
        "weight": 1.5,
        "opacity": 0.5,
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=["name", "crash_count"],
        aliases=["Borough:", "Crashes:"],
        sticky=True,
        delay=500
    )
)
borough_layer.add_to(nyc_map)

# Prepare heatmap data
heatmap_data = df[['LATITUDE_CRASH', 'LONGITUDE_CRASH']].dropna().values.tolist()

# Add Heatmap Layer
HeatMap(heatmap_data, radius=10, blur=10, min_opacity=0.2).add_to(nyc_map)

# Add borough labels at their centroids
for _, row in gdf_boroughs.iterrows():
    borough_name = row["name"]
    centroid = row["geometry"].centroid  # Get centroid of the borough
    crash_count = borough_crash_counts.get(borough_name, 0)

    # Add text label at the centroid with borough name
    folium.Marker(
        location=[centroid.y + 0.02, centroid.x - 0.02],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; font-weight: bold; color: black;">{borough_name}</div>'),
    ).add_to(nyc_map)

# Save the interactive map
map_filename = 'output/NYC_Accident_Heatmap_with_Boroughs.html'
nyc_map.save(map_filename)

# Print the map link
print(f'NYC Accident Heatmap with Borough Boundaries: {map_filename}')


########## 2.2.4 Bar graph for comparing crash counts across boroughs #########################
crash_counts = df["BOROUGH"].value_counts().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=crash_counts.index, y=crash_counts.values, color="steelblue")

# Add value labels on top of bars
for i, val in enumerate(crash_counts.values):
    plt.text(i, val + 100, str(val), ha='center', va='bottom', fontsize=9)

plt.title("Number of Crashes by Borough")
plt.xlabel("Borough")
plt.ylabel("Crash Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/Number of Crashes by Borough.png", dpi=300)


# ####################### 2.3. Generate bar graph and choropleth map shows crash severity based on zip code ########################################
# Reference 1: Manual on Classification of Motor Vehicle Traffic Crashes, NHTSA
# URL: https://www.nhtsa.gov/sites/nhtsa.gov/files/documents/ansi_d16-2017.pdf
# Reference 2: Crash Costs for Highway Safety Analysis
# URL: https://rosap.ntl.bts.gov/view/dot/42858

# Convert ZIP_CODE to standardized format
df["ZIP_CODE"] = df["ZIP_CODE"].astype(str).str.split('.').str[0].str.zfill(5)
# print(df.head())

# Define a function to classify crash severity levels
def classify_severity(row):
    total_killed = (
            row["NUMBER_OF_PERSONS_KILLED"] +
            row["NUMBER_OF_PEDESTRIANS_KILLED"] +
            row["NUMBER_OF_CYCLIST_KILLED"] +
            row["NUMBER_OF_MOTORIST_KILLED"]
    )

    total_injured = (
            row["NUMBER_OF_PERSONS_INJURED"] +
            row["NUMBER_OF_PEDESTRIANS_INJURED"] +
            row["NUMBER_OF_CYCLIST_INJURED"] +
            row["NUMBER_OF_MOTORIST_INJURED"]
    )

    if total_killed >= 3:
        return "Fatal (>= 3 killed)"
    elif 0 < total_killed < 3:
        return "Fatal (< 3 killed)"
    elif total_injured >= 3:
        return "Injury (>= 3 injured)"
    elif 0 < total_injured < 3:
        return "Injury (< 3 injured)"
    else:
        return "Property Damage Only"

# Apply severity classification
df["CRASH_SEVERITY"] = df.apply(classify_severity, axis=1)
# print(df.head())

# Aggregate crash severity counts per ZIP code
severity_counts = df.groupby(["ZIP_CODE", "CRASH_SEVERITY"]).size().unstack(fill_value=0)

# Save processed data
severity_counts_path = "output/crash_severity_by_zip.csv"
severity_counts.to_csv(severity_counts_path)

################# 2.3.1 Bar graph showing crash cases by severity level ##############
severity_df = pd.read_csv(severity_counts_path, index_col=0)
# print(severity_df)

# Sum crash cases across all ZIP codes for each severity level
severity_totals = severity_df.sum().sort_values(ascending=False)

# Plot bar graph for severity distribution
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.barplot(x=severity_totals.index, y=severity_totals.values, color="steelblue")

# Add value labels on top of bars
for i, val in enumerate(severity_totals.values):
    plt.text(i, val + 100, str(val), ha='center', va='bottom', fontsize=9)

plt.title("Crash Cases by Severity Level")
plt.xlabel("Crash Severity")
plt.ylabel("Number of Crashes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/crash_severity_distribution.png", dpi=300)

# ############### 2.3.2. Choropleth map shows crash severity based on zip code ##############

# Load the NYC ZIP Code GeoJSON file
nyc_zip_geojson_path = "nyc-zip-code-tabulation-areas-polygons.geojson"
with open(nyc_zip_geojson_path, "r") as f:
    nyc_zip_geojson = json.load(f)

# Ensure ZIP codes in severity data are strings
severity_df.index = severity_df.index.astype(str).str.zfill(5)

# Compute a severity score per ZIP code using a weighted sum
severity_weights = {
    "Property Damage Only": 1,
    "Injury (< 3 injured)": 2,
    "Injury (>= 3 injured)": 3,
    "Fatal (< 3 killed)": 4,
    "Fatal (>= 3 killed)": 5
}

# Compute severity score and normalized score
severity_df["Severity_Score"] = sum(
    severity_df[col] * severity_weights[col] for col in severity_weights
)
severity_df["Severity_Score_Norm"] = severity_df["Severity_Score"] / severity_df["Severity_Score"].max() * 100
# print(severity_df)

# Add normalized severity score to GeoJSON properties
for feature in nyc_zip_geojson["features"]:
    zip_code = str(feature["properties"].get("postalCode", "")).zfill(5)
    score = severity_df["Severity_Score_Norm"].get(zip_code, 0)
    feature["properties"]["severity_score"] = round(score, 4)

# Create a Choropleth Map
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="cartodb positron")

# Define color scale
colormap = branca.colormap.LinearColormap(
    colors=["green", "yellow", "orange", "red", "darkred"],
    vmin=0,
    vmax=severity_df["Severity_Score_Norm"].max(),
    caption="Crash Severity Level"
)

# Add ZIP Code polygons with severity scores
folium.GeoJson(
    nyc_zip_geojson,
    name="Crash Severity by ZIP Code",
    style_function=lambda feature: {
        "fillColor": colormap(feature["properties"]["severity_score"]),
        "color": "black",
        "weight": 0.5,
        "opacity": 0.7,
        "fillOpacity": 0.6,
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=["postalCode", "severity_score"],
        aliases=["ZIP Code:", "Severity Score (Norm):"],
        sticky=True,
        delay=500
    )
).add_to(nyc_map)

# Add color legend
colormap.add_to(nyc_map)

# Save the interactive map
map_filename = "output/NYC_Crash_Severity_Choropleth.html"
nyc_map.save(map_filename)

# Print the map link
print(f'Choropleth Map showing crash severity by ZIP code: {map_filename}')


# ####################################################################################################################################################
# # 3. Weather impact on accidents to how weather conditions affect crashes.

# # 3.1. Horizontal bar graphs for PRCP, SNWD, SNWD, TMAX, TMIN, SEASON

# Define bin settings specific for PRCP, SNWD, SNWD, TMAX, TMIN
weather_vars = ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]

bin_settings = {
    "PRCP": {"range": [1, 10], "bins": 9},
    "SNOW": {"range": [5, 35], "bins": 6},
    "SNWD": {"range": [5, 35], "bins": 6},
    "TMAX": {"range": [-20, 120], "bins": 14},
    "TMIN": {"range": [-20, 100], "bins": 12}
}

# Define units for y-axis labels
weather_units = {
    "PRCP": " (inches)",
    "SNOW": " (inches)",
    "SNWD": " (inches)",
    "TMAX": " (°F)",
    "TMIN": " (°F)"
}

# Create figure with 3 rows, 2 columns to include SEASON as the 6th plot
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))
axes = axes.flatten()

for i, var in enumerate(weather_vars):
    if var in df.columns:
        subset = df[[var]].dropna()

        if var in ["TMAX", "TMIN"]:
            subset[var] = subset[var].round(0)

        # Define custom bin edges
        bin_info = bin_settings[var]
        bin_edges = pd.interval_range(
            start=bin_info["range"][0],
            end=bin_info["range"][1],
            periods=bin_info["bins"]
        )
        subset["binned"] = pd.cut(subset[var], bins=bin_edges)
        count_by_bin = subset["binned"].value_counts().sort_index()

        sns.barplot(
            y=count_by_bin.index.astype(str),
            x=count_by_bin.values,
            color="steelblue",
            ax=axes[i]
        )
        axes[i].set_title(f"Number of Accidents by {var} Bins")
        axes[i].set_xlabel("Number of Accidents")
        axes[i].set_ylabel(f"{var}{weather_units.get(var, '')}")

# Add horizontal bar chart for SEASON
season_counts = df["SEASON"].value_counts().reindex(["Winter", "Spring", "Summer", "Fall"])
sns.barplot(
    y=season_counts.index,
    x=season_counts.values,
    color="steelblue",
    ax=axes[-1]
)
axes[-1].set_title("Number of Accidents by Season")
axes[-1].set_xlabel("Number of Accidents")
axes[-1].set_ylabel("Season")

plt.tight_layout()
plt.savefig("output/accidents_vs_weather_barcharts.png", dpi=300)


# # 3.2. Stacked bar chart bar graph for Seasonal and vehicle types
combined_vehicle_df = pd.concat([
    df[["SEASON", "VEHICLE_TYPE_CODE_1_REGROUP"]].rename(columns={"VEHICLE_TYPE_CODE_1_REGROUP": "VEHICLE"}),
    df[["SEASON", "VEHICLE_TYPE_CODE_2_REGROUP"]].rename(columns={"VEHICLE_TYPE_CODE_2_REGROUP": "VEHICLE"})
])

# Drop rows with missing vehicle types
combined_vehicle_df = combined_vehicle_df.dropna()

# Group and pivot to prepare for stacked bar chart
grouped_combined = combined_vehicle_df.groupby(["SEASON", "VEHICLE"]).size().reset_index(name="count")
pivot_combined = grouped_combined.pivot(index="SEASON", columns="VEHICLE", values="count").fillna(0)

# Reorder seasons
season_order = ["Winter", "Spring", "Summer", "Fall"]
pivot_combined = pivot_combined.reindex(season_order)

# Select top 6 vehicle types overall
top_combined_vehicle_types = pivot_combined.sum().sort_values(ascending=False).head(6).index
pivot_combined_top = pivot_combined[top_combined_vehicle_types]

# Plot and save the stacked bar chart
ax = pivot_combined_top.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab20")
plt.title("Number of Accidents by Season and Vehicle Type (Combined)")
plt.xlabel("Season")
plt.ylabel("Number of Accidents")
plt.legend(title="Vehicle Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("output/season_combined_vehicletype_stacked_bar.png")


# # 3.3. Stacked bar chart for Month and season
# Group data by SEASON and MONTH
season_month_grouped = df.groupby(["SEASON", "MONTH"]).size().reset_index(name="count")

# Pivot for stacked bar chart (SEASON on X, MONTH as stacked layers)
pivot_season_month = season_month_grouped.pivot(index="SEASON", columns="MONTH", values="count").fillna(0)

# Reorder seasons
pivot_season_month = pivot_season_month.reindex(season_order)

# Plot and save the stacked bar chart
ax = pivot_season_month.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab20c")
plt.title("Number of Accidents by Season and Month")
plt.xlabel("Season")
plt.ylabel("Number of Accidents")
plt.legend(title="Month", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("output/season_month_stacked_bar.png")





