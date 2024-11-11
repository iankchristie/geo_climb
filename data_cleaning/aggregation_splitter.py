import pandas as pd
from sklearn.model_selection import train_test_split


def split_csv_into_splits(csv_filepath: str):
    df = pd.read_csv(csv_filepath)

    # Shuffle and split the data: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv("data/training.csv", index=False)
    val_df.to_csv("data/validation.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("Data split into training.csv, validation.csv, and test.csv")


if __name__ == "__main__":
    split_csv_into_splits("data/aggregation.csv")
