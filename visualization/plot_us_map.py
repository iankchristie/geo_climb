import sys
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloaders.file_utils import *


def plot_geo_points_us(lat_lons: set[tuple[float, float]]):
    _, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    for lat, lon in lat_lons:
        ax.plot(
            lon, lat, marker="o", color="blue", markersize=5, transform=ccrs.Geodetic()
        )

    # Set the extent of the map to focus on the US
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True)
    plt.title("Unlabeled Locations")
    plt.show()


if __name__ == "__main__":
    labeled_lat_lons = set(
        get_undownloaded_lat_lons("data/unlabeled/unlabeled_locations.csv")
    )
    plot_geo_points_us(labeled_lat_lons)
    # unlabeled_lat_lons = set(
    #     get_undownloaded_lat_lons("data/unlabeled/unlabeled_locations.csv")
    # )
    # plot_geo_points_us(unlabeled_lat_lons)
