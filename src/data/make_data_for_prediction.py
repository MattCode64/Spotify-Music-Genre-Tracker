import pandas as pd
import umap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


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


def make_data(input_data):
    # # Read 1st line of a csv file
    # input_data = {
    #     "popularity": 73,
    #     "duration_ms": 230666,
    #     "explicit": False,
    #     "danceability": 0.676,
    #     "energy": 0.461,
    #     "key": 1,
    #     "loudness": -6.746,
    #     "mode": 0,
    #     "speechiness": 0.143,
    #     "acousticness": 0.0322,
    #     "instrumentalness": 1.01e-06,
    #     "liveness": 0.358,
    #     "valence": 0.715,
    #     "tempo": 87.917,
    #     "time_signature": 4,
    #     "track_genre": "acoustic"
    # }

    # Convert dictionary to DataFrame
    data = pd.DataFrame([input_data])
    # Data cleaning and processing
    data = replace_boolean(data, 'explicit')
    data = label_encode(data, 'track_genre')

    columns_to_keep = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
                       'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                       'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
                       'track_genre_encoded']
    data = data[columns_to_keep]

    data = standardize_columns(data, data.drop(columns=['track_genre_encoded']).columns)

    data_ready_for_prediction = apply_umap(data, 'track_genre_encoded', n_components=6)

    # print(data_ready_for_prediction)
    return data_ready_for_prediction


def prepare_data_df(input_df):
    # Data cleaning and processing
    input_df = replace_boolean(input_df, 'explicit')
    input_df = label_encode(input_df, 'track_genre')

    columns_to_keep = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
                       'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                       'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
                       'track_genre_encoded']
    input_df = input_df[columns_to_keep]

    input_df = standardize_columns(input_df, input_df.drop(columns=['track_genre_encoded']).columns)

    data_ready_for_prediction = apply_umap(input_df, 'track_genre_encoded', n_components=6)

    return data_ready_for_prediction


if __name__ == '__main__':
    main()
