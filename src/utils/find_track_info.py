import pandas as pd


def get_track_info(track_id, csv_a_path, csv_b_path):
    """
    Retrieve track information based on the provided track_id.

    Parameters:
        track_id (str): The unique identifier for the track.
        csv_a_path (str): Path to CSV A, containing track_id.
        csv_b_path (str): Path to CSV B, containing track_id, 'artists', 'track_name', and 'track_genre'.

    Returns:
        dict: A dictionary with 'artists', 'track_name', and 'track_genre' if track_id is found in both CSVs.
        None: If the track_id is not found in CSV B.
    """
    # Load CSV files
    df_a = pd.read_csv(csv_a_path)
    df_b = pd.read_csv(csv_b_path)

    # Check if track_id is in CSV A
    if track_id not in df_a['track_id'].values:
        return None

    # Find the row in CSV B corresponding to the track_id
    track_info = df_b[df_b['track_id'] == track_id]

    if track_info.empty:
        return None

    # Extract relevant information
    track_info = track_info.iloc[0]  # Get the first row as a Series
    return {
        'artists': track_info['artists'],
        'track_name': track_info['track_name'],
        'track_genre': track_info['track_genre']
    }


def main():
    print(f"Starting {main.__name__}")
    csv_a = "/home/matthieu/UbuntuData/PycharmProjects/MelodAI/data/raw/spotify_tracks.csv"
    csv_b = "/home/matthieu/UbuntuData/PycharmProjects/MelodAI/data/raw/dataset.csv"

    track_id = "4uUG5RXrOk84mYEfFvj3cK"

    track_info = get_track_info(track_id, csv_a, csv_b)
    if track_info is None:
        print(f"Track ID {track_id} not found.")
    else:
        print(f"Track ID: {track_id}")
        print(f"Artists: {track_info['artists']}")
        print(f"Track Name: {track_info['track_name']}")
        print(f"Track Genre: {track_info['track_genre']}")


if __name__ == '__main__':
    main()
