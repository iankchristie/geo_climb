import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS

# Read the CSV file
file_path = "data/labeled/climbing_locations.csv"
data = pd.read_csv(file_path)

# Convert latitude and longitude into GeoDataFrame
gdf = gpd.GeoDataFrame(
    data,
    geometry=[Point(lon, lat) for lat, lon in zip(data["Latitude"], data["Longitude"])],
    crs=CRS.from_epsg(4326),  # WGS84 coordinate system
)

# Reproject to a projected CRS (e.g., UTM zone 11N for western US)
gdf = gdf.to_crs(epsg=26911)

# Buffer each point by 1 km (1000 meters)
gdf["buffer"] = gdf.geometry.buffer(1000)

# Merge the buffers into a single geometry
merged_geometry = gdf["buffer"].unary_union

# Calculate the total area in square meters
total_area = merged_geometry.area
print(f"Total Area: {total_area / 1e6:.2f} square kilometers")
