import sys
import os
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_utils import *
import json


def format_text(data):
    formatted_lines = []
    for lith in data["liths"]:
        lith_id = lith.get("lith_id", "N/A")
        lith_name = lith.get("lith", "N/A")
        lith_class = lith.get("lith_class", "N/A")
        lith_type = lith.get("lith_type", "N/A")

        formatted_line = (
            f"ID: {lith_id}, Lith: {lith_name}, Class: {lith_class}, Type: {lith_type}"
        )
        formatted_lines.append(formatted_line)

    return "\n".join(formatted_lines)


def view_lithology(
    data_dir: str,
    ax: Axes,
    latitude: float = 40.0150,
    longitude: float = -105.2705,
):
    data = {}
    output_json = encode_file(latitude, longitude, "lit", data_dir, file_type="json")
    with open(output_json, "r") as json_file:
        data = json.load(json_file)
        data = data["success"]["data"]["mapData"][0]
        if not data:
            data["lith"] = ""
            data["liths"] = []
        print(json.dumps(data, indent=4))

    ax.text(
        0.5,
        0.5,
        format_text(data),
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
    )
    ax.axis("off")
