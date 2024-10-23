import sys
import os
import random
import numpy as np
from sklearn.neighbors import BallTree
import geopandas as gpd
from shapely.geometry import Point

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from downloaders.file_utils import *
from visualization.plot_us_map import *

EARTH_RADIUS_KM = 6371.0


def load_us_shape():
    # https://stackoverflow.com/questions/74378025/generate-random-coordinates-in-united-states
    url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_nation_20m.zip"
    USA = gpd.read_file(url).explode()

    USA = USA.loc[USA.geometry.apply(lambda x: x.exterior.bounds[2]) < -60]

    return USA


def generate_random_latlon(geo_shape):
    while True:
        minx, miny, maxx, maxy = geo_shape.total_bounds
        lat = random.uniform(miny, maxy)
        lon = random.uniform(minx, maxx)

        point = Point(lon, lat)
        if geo_shape.contains(point).any():
            return lat, lon


def is_far_enough(
    ball_tree: BallTree, new_lat_lon: tuple[float, float], min_distance_km: float
) -> bool:
    # Ball Tree uses radians
    new_lat_lon_rad = np.radians([new_lat_lon])

    dist, _ = ball_tree.query(new_lat_lon_rad, k=1)

    # Convert distance from radians to kilometers
    dist_km = dist[0][0] * EARTH_RADIUS_KM

    return dist_km >= min_distance_km


def generate_lat_lons(
    lat_lons: list[tuple[float, float]], num: int, min_distance_km: int = 1
) -> set[tuple[float, float]]:
    new_lat_lons = set()

    # Ball Tree uses radians
    lat_lons_rad = np.radians(np.array(lat_lons))
    ball_tree = BallTree(lat_lons_rad, metric="haversine")

    us_boundary_gdf = load_us_shape()

    while len(new_lat_lons) < num:
        new_lat_lon = generate_random_latlon(us_boundary_gdf)

        if is_far_enough(ball_tree, new_lat_lon, min_distance_km):
            new_lat_lons.add(new_lat_lon)

            if len(new_lat_lons) % 20 == 0:
                print(f"{len(new_lat_lons)} locations generated")

    return new_lat_lons


if __name__ == "__main__":
    labeled_lat_lons = get_undownloaded_lat_lons()
    unlabeled_lat_lons = generate_lat_lons(
        labeled_lat_lons, num=len(labeled_lat_lons), min_distance_km=2
    )
    plot_geo_points_us(unlabeled_lat_lons)
    save_lat_lons_to_csv(unlabeled_lat_lons, "data/unlabeled/unlabled_locations.csv")
