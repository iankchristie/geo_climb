import pandas as pd


def filter_csv(
    input_file_path: str,
    output_file_path: str,
):
    df = pd.read_csv(input_file_path)

    df_filtered = df[
        ~df["Location"].str.strip().str.endswith(("International", "Alaska", "Hawaii"))
    ]

    print(df_filtered.columns)

    columns_to_keep = [
        "Avg Stars",
        "Pitches",
        "Length",
        "Area Latitude",
        "Area Longitude",
        "num_votes",
    ]
    filtered_df = df_filtered[columns_to_keep]

    filtered_df.to_csv(output_file_path, index=False)

    print(f"Filtered data written to {output_file_path}")


def write_unique_lat_lons(
    input_file_path: str,
    output_file_path: str,
) -> None:
    df = pd.read_csv(input_file_path)

    # Select latitude and longitude columns and drop rows with missing values
    lat_lon_pairs = df[["Area Latitude", "Area Longitude"]].dropna()

    unique_lat_lons = lat_lon_pairs.drop_duplicates()
    unique_lat_lons = unique_lat_lons.round(5)

    unique_lat_lons.to_csv(
        output_file_path, index=False, header=["Latitude", "Longitude"]
    )

    print(f"Unique latitude and longitude pairs written to {output_file_path}")


if __name__ == "__main__":
    filter_csv(
        input_file_path="./data/labeled/climbing/mp_routes.csv",
        output_file_path="./data/labeled/climbing/filtered_mtp.csv",
    )
    write_unique_lat_lons(
        input_file_path="./data/labeled/climbing/filtered_mtp.csv",
        output_file_path="./data/labeled/climbing_locations.csv",
    )
