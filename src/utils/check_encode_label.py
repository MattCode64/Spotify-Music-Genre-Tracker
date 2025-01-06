import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_track_genre(input_csv, output_csv):
    # Load the data
    df = pd.read_csv(input_csv)

    # Check if 'track_genre' column exists
    if 'track_genre' not in df.columns:
        raise ValueError("The input CSV file must contain a 'track_genre' column.")

    # Encode the 'track_genre' column
    encoder = LabelEncoder()
    df['track_genre_encoded'] = encoder.fit_transform(df['track_genre'])

    # Drop duplicate rows
    df = df.drop_duplicates(subset=['track_genre', 'track_genre_encoded'])

    # Save the encoded data to a new CSV file
    df[['track_genre', 'track_genre_encoded']].to_csv(output_csv, index=False)
    print(f"Encoded data saved to {output_csv}")


# Example usage
input_csv = '/home/matthieu/UbuntuData/PycharmProjects/MelodAI/data/raw/dataset.csv'
output_csv = '/home/matthieu/UbuntuData/PycharmProjects/MelodAI/reports/encoded_dataset.csv'
encode_track_genre(input_csv, output_csv)
