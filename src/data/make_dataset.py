import os

import pandas as pd
import umap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from src.utils.dagshub_interactions import upload_file_to_dagshub

# Dynamically determine the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw/dataset.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/final_dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/model.pkl")


# Stage: process_data
def load_data(path):
    return pd.read_csv(path, low_memory=False)


def label_encode(df, column):
    encoder = LabelEncoder()
    df[column + '_encoded'] = encoder.fit_transform(df[column])
    return df


def replace_boolean(df, column):
    df[column] = df[column].replace({True: 1, False: 0})
    df[column] = df[column].infer_objects(copy=False)
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


def apply_umap(df, target_column, n_components=2):
    umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
    df_std_scaler = df.drop(columns=[target_column])
    reduced = umap_reducer.fit_transform(df_std_scaler)

    # Create a DataFrame with UMAP components
    df_umap = pd.DataFrame(reduced, columns=[f'UMAP{i + 1}' for i in range(n_components)])

    # Concatenate with the target column
    df_umap = pd.concat([df_umap, df[target_column].reset_index(drop=True)], axis=1)

    return df_umap


def save_data_locally(df, path):
    df.to_csv(path, index=False)


from sklearn.model_selection import train_test_split


def split_datasets(X, y,
                   test_size=0.2,
                   val_size=None,
                   stratify=True,
                   random_state=42):
    """
    Splits data into train/test or train/val/test sets.

    Parameters:
        X (pd.DataFrame): Features dataframe.
        y (pd.Series): Target series.
        test_size (float): Proportion of the data to allocate to the test set.
        val_size (float, optional): Proportion of the training data to allocate to the validation set.
        stratify (bool): Whether to use stratified sampling based on `y`.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Returns the splits as tuples:
              (X_train, X_test, y_train, y_test) or
              (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Ensure stratification is handled
    stratify_param = y if stratify else None

    # Initial split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_param, random_state=random_state
    )

    if val_size:
        # Further split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, stratify=y_train if stratify else None, random_state=random_state
        )

    # Return datasets as tuple
    if val_size:
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test


def main():
    # Load raw data
    df = load_data(RAW_DATA_PATH)
    df = drop_missing_values(df)

    # Data cleaning and processing
    df = replace_boolean(df, 'explicit')
    df = label_encode(df, 'track_genre')
    df = drop_missing_values(df)

    columns_to_drop = ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre']
    df = drop_columns(df, columns_to_drop)

    df = standardize_columns(df, df.drop(columns=['track_genre_encoded']).columns)

    # Apply UMAP for dimensionality reduction
    df = apply_umap(df, 'track_genre_encoded', n_components=6)

    # TODO: Data validation

    # Save processed data
    save_data_locally(df, PROCESSED_DATA_PATH)

    # Upload processed data to DagsHub
    upload_file_to_dagshub("MattCode64/MelodAI", "MelodAI", PROCESSED_DATA_PATH, "final_dataset.csv")


if __name__ == '__main__':
    main()
