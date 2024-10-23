import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_geo_points_us(lat_lons: set[tuple[float, float]]):
    _, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    for lat, lon in lat_lons:
        ax.plot(
            lon, lat, marker="o", color="red", markersize=5, transform=ccrs.Geodetic()
        )

    # Set the extent of the map to focus on the US
    ax.set_extent([-130, -60, 20, 55], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True)
    plt.title("Generated Latitude/Longitude Points")
    plt.show()
