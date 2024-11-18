import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Append the root directory of your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config


def split_csv_into_splits(csv_filepath: str):
    df = pd.read_csv(csv_filepath)

    # Shuffle and split the data: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv("data/training_split.csv", index=False)
    val_df.to_csv("data/validation_split.csv", index=False)
    test_df.to_csv("data/test_split.csv", index=False)
    print("Data split into training.csv, validation.csv, and test.csv")


if __name__ == "__main__":
    split_csv_into_splits("data/all_lat_lons.csv")
