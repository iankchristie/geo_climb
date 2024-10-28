import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_max_pitches_per_location(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}")
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
        return

    # Check if the required columns exist
    required_columns = {"Pitches", "Area Latitude", "Area Longitude"}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        print(f"Error: Missing columns in the CSV file: {', '.join(missing_cols)}")
        return

    # Group by 'Area Latitude' and 'Area Longitude' and get the max 'Pitches' for each group
    max_pitches_per_location = (
        df.groupby(["Area Latitude", "Area Longitude"])["Pitches"].max().reset_index()
    )
    print("Calculated maximum number of pitches at each unique location.")

    # Calculate the distribution of the maximum pitches
    pitches_distribution = (
        max_pitches_per_location["Pitches"].value_counts().sort_index()
    )
    print("Pitches distribution calculated successfully.")

    num_routes_1_pitch = max_pitches_per_location[
        max_pitches_per_location["Pitches"] == 1
    ].shape[0]
    num_routes_2_or_more_pitches = max_pitches_per_location[
        max_pitches_per_location["Pitches"] >= 2
    ].shape[0]

    print(num_routes_1_pitch)
    print(num_routes_2_or_more_pitches)
    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    pitches_distribution.plot(kind="bar", color="skyblue", edgecolor="black")

    # Set plot titles and labels
    plt.title("Distribution of Maximum Number of Pitches per Location", fontsize=16)
    plt.xlabel("Maximum Number of Pitches", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    # Rotate x-ticks if necessary
    plt.xticks(rotation=0)

    # Add gridlines for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_max_avg_stars_per_location(csv_path):
    """
    Opens a CSV file at the given path and performs the following:
    - Calculates the maximum 'Avg Stars' at each unique latitude and longitude.
    - Bins the 'Avg Stars' into intervals for plotting.
    - Plots a histogram of the distribution of the maximum 'Avg Stars' per location.
    - Prints out the number of routes with 'Avg Stars' in different ranges.

    Parameters:
    csv_path (str): The file path to the CSV file.

    The CSV file is expected to have the following headings:
    'Avg Stars', 'Pitches', 'Length', 'Area Latitude', 'Area Longitude', 'num_votes'
    """

    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}")
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
        return

    # Check if the required columns exist
    required_columns = {"Avg Stars", "Area Latitude", "Area Longitude"}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        print(f"Error: Missing columns in the CSV file: {', '.join(missing_cols)}")
        return

    # Ensure 'Avg Stars' column is numeric
    df["Avg Stars"] = pd.to_numeric(df["Avg Stars"], errors="coerce")

    # Drop rows with NaN values in 'Avg Stars', 'Area Latitude', or 'Area Longitude'
    df.dropna(subset=["Avg Stars", "Area Latitude", "Area Longitude"], inplace=True)

    # Group by 'Area Latitude' and 'Area Longitude' and get the max 'Avg Stars' for each group
    max_avg_stars_per_location = (
        df.groupby(["Area Latitude", "Area Longitude"])["Avg Stars"].max().reset_index()
    )
    print("Calculated maximum 'Avg Stars' at each unique location.")

    # Define bins for 'Avg Stars'
    bins = np.arange(0, 5.5, 0.5)  # Bins from 0 to 5 in steps of 0.5

    # Bin the 'Avg Stars' values
    max_avg_stars_per_location["Stars Binned"] = pd.cut(
        max_avg_stars_per_location["Avg Stars"], bins=bins, right=False
    )
    stars_distribution = (
        max_avg_stars_per_location["Stars Binned"].value_counts().sort_index()
    )
    print("Stars distribution calculated successfully.")

    # **Additional Task: Print the number of routes with 'Avg Stars' in different ranges**
    num_routes_below_3 = max_avg_stars_per_location[
        max_avg_stars_per_location["Avg Stars"] < 3.0
    ].shape[0]
    num_routes_3_to_4 = max_avg_stars_per_location[
        (max_avg_stars_per_location["Avg Stars"] >= 3.0)
        & (max_avg_stars_per_location["Avg Stars"] < 4.0)
    ].shape[0]
    num_routes_4_and_above = max_avg_stars_per_location[
        max_avg_stars_per_location["Avg Stars"] >= 4.0
    ].shape[0]

    total_routes = num_routes_below_3 + num_routes_3_to_4 + num_routes_4_and_above

    print(f"\nNumber of routes with Avg Stars below 3.0: {num_routes_below_3}")
    print(f"Number of routes with Avg Stars between 3.0 and 4.0: {num_routes_3_to_4}")
    print(f"Number of routes with Avg Stars 4.0 and above: {num_routes_4_and_above}")
    print(f"Total number of routes: {total_routes}\n")

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    stars_distribution.plot(kind="bar", color="goldenrod", edgecolor="black")

    # Set plot titles and labels
    plt.title("Distribution of Maximum Avg Stars per Location", fontsize=16)
    plt.xlabel("Average Stars (Binned)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    # Rotate x-ticks for better readability
    plt.xticks(rotation=45)

    # Add gridlines for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # plot_max_pitches_per_location("data/labeled/climbing/filtered_mtp.csv")
    plot_max_avg_stars_per_location("data/labeled/climbing/filtered_mtp.csv")
