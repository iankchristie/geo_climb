import pandas as pd


def filter_csv(
    input_file_path="./data/scratch/mtp.csv",
    output_file_path="./data/scratch/filtered_mtp.csv",
):
    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Filter rows where 'Location' does not end with "International"
    df_filtered = df[~df["Location"].str.strip().str.endswith("International")]

    print(df_filtered.columns)

    # Select only the columns you need
    columns_to_keep = [
        "Avg Stars",
        "Pitches",
        "Length",
        "Area Latitude",
        "Area Longitude",
        "num_votes",
    ]
    filtered_df = df_filtered[columns_to_keep]

    # Write the filtered data to a new CSV file
    filtered_df.to_csv(output_file_path, index=False)

    print(f"Filtered data written to {output_file_path}")


def write_unique_lat_lons(input_file: str, output_file: str) -> None:
    """
    Reads lat/lon pairs from the input CSV file, deduplicates them,
    and writes the unique pairs to the output file.

    Parameters:
    - input_file: Path to the input CSV file.
    - output_file: Path to the output file where unique lat/lon pairs will be written.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Select latitude and longitude columns and drop rows with missing values
    lat_lon_pairs = df[["Area Latitude", "Area Longitude"]].dropna()

    # Deduplicate the latitude and longitude pairs
    unique_lat_lons = lat_lon_pairs.drop_duplicates()

    # Write the unique lat/lon pairs to a new CSV file
    unique_lat_lons.to_csv(output_file, index=False, header=["Latitude", "Longitude"])

    print(f"Unique latitude and longitude pairs written to {output_file}")


if __name__ == "__main__":
    input_file = "./data/scratch/filtered_mtp.csv"
    output_file = "./data/scratch/lat_lons.csv"
    write_unique_lat_lons(input_file, output_file)
