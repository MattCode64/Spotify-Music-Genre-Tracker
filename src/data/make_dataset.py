import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Dynamically determine the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw/dataset.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/final_dataset.csv")


def load_data(dataset_path):
    return pd.read_csv(dataset_path, low_memory=False)


def label_encode(df, column):
    label_encoder = LabelEncoder()
    df[column + '_encoded'] = label_encoder.fit_transform(df[column])
    return df


def replace_boolean(df, column):
    df[column] = df[column].replace({True: 1, False: 0})
    return df


def drop_missing_values(df):
    df.dropna(inplace=True)
    return df


def drop_columns(df, columns):
    df.drop(columns=columns, inplace=True)
    return df


def standardize_columns(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def save_data(df, path):
    df.to_csv(path, index=False)


def select_features(df, cols):
    try:
        selected_df = df[cols]
    except KeyError as e:
        print(f"Error: One or more columns not found in the dataframe: {e}")
        return None
    return selected_df


def main():
    dataset_path = DATA_PATH
    # preprocessed_dataset_path = "../data/processed/preprocessed_dataset.csv"
    final_dataset_path = PROCESSED_DATA_PATH

    # Load data
    df = load_data(dataset_path)

    # Label encode 'track_genre'
    df = label_encode(df, 'track_genre')

    # Replace boolean values in 'explicit'
    df = replace_boolean(df, 'explicit')

    # Drop missing values
    df = drop_missing_values(df)

    # Drop unnecessary columns
    columns_to_drop = ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre']
    df = drop_columns(df, columns_to_drop)

    # # Save preprocessed data
    # save_data(df, preprocessed_dataset_path)
    #
    # # Load preprocessed data
    # df = load_data(preprocessed_dataset_path)

    # Standardize numerical columns
    col_to_standardize = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'loudness', 'speechiness',
                          'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    df = standardize_columns(df, col_to_standardize)

    # Select specific features
    selected_columns = ['danceability', 'energy', 'loudness', 'instrumentalness', 'valence', 'track_genre_encoded']
    df = select_features(df, selected_columns)

    # Save final dataset
    if df is not None:
        save_data(df, final_dataset_path)

    # Save final dataset
    save_data(df, final_dataset_path)


if __name__ == "__main__":
    main()
