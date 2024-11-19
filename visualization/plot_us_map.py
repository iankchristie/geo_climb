import sys
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_utils import *
from visualization.side_by_side import plot_data


def on_click(event, lat_lons, labeled):
    """
    Callback function to handle mouse click events.
    Displays the latitude and longitude of the clicked point, only if near a data point.
    """
    # Check if the click was within the axes
    if not event.inaxes:
        return

    # Get the clicked coordinates
    clicked_lon, clicked_lat = event.xdata, event.ydata

    # Define a tolerance (in degrees) for detecting clicks near data points
    tolerance = 0.5  # Adjust this value as needed

    # Check if the click is near any of the plotted points
    for lat, lon in lat_lons:
        if abs(clicked_lat - lat) <= tolerance and abs(clicked_lon - lon) <= tolerance:
            print(f"Clicked on data point at Latitude: {lat:.4f}, Longitude: {lon:.4f}")
            plot_data(lat, lon, labeled)
            break


def plot_geo_points_us(lat_lons: set[tuple[float, float]], labeled: bool):
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    for lat, lon in lat_lons:
        ax.plot(
            lon, lat, marker="o", color="blue", markersize=3, transform=ccrs.Geodetic()
        )

    # Set the extent of the map to focus on the US
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True)
    plt.title("Unlabeled Locations")

    fig.canvas.mpl_connect(
        "button_press_event", lambda event: on_click(event, lat_lons, labeled)
    )
    plt.show()


if __name__ == "__main__":
    labeled_lat_lons = set(
        get_undownloaded_lat_lons("data/labeled/climbing_locations.csv")
    )
    plot_geo_points_us(labeled_lat_lons, True)
    # unlabeled_lat_lons = set(
    #     get_undownloaded_lat_lons("data/unlabeled/unlabeled_locations.csv")
    # )
    # plot_geo_points_us(unlabeled_lat_lons, False)
