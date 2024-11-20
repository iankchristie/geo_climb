from sklearn.neighbors import BallTree
import numpy as np


class LatLonTree:
    def __init__(self, lat_lons: list[tuple[float, float]]) -> None:
        # Convert lat-lons to radians for haversine distance
        self.lat_lons = lat_lons
        lat_lons_rad = np.radians(np.array(lat_lons))
        self.ball_tree = BallTree(lat_lons_rad, metric="haversine")

    def query(self, new_lat_lon: tuple[float, float]):
        # Convert new point to radians and query the nearest neighbor
        new_lat_lon_rad = np.radians([new_lat_lon])
        distances, indices = self.ball_tree.query(new_lat_lon_rad, k=1)
        # Convert distance from radians to kilometers
        distance_km = distances[0][0] * 6371  # Radius of Earth in km
        nearest_lat_lon = self.lat_lons[indices[0][0]]
        return distance_km, nearest_lat_lon
