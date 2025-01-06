import mlflow
import pandas as pd

# Configurer le tracking URI distant
DAGSHUB_URI = "https://dagshub.com/MattCode64/MelodAI.mlflow"
mlflow.set_tracking_uri(DAGSHUB_URI)

try:
    # Charger le modèle distant
    model_uri = "runs:/6f31d1a4f3074535805b56933e52ead4/Artifacts"
    model = mlflow.pyfunc.load_model(model_uri)
    print("Modèle chargé avec succès.")

    # Tester une prédiction
    data = pd.DataFrame({
        "UMAP1": [8.4346075],
        "UMAP2": [-2.5405962],
        "UMAP3": [-2.6248655],
        "UMAP4": [7.920587],
        "UMAP5": [-0.8108143],
        "UMAP6": [6.02044]
    })  # Track genre : 41
    predictions = model.predict(data)
    print(f"Prédictions : {predictions}")

    track_genre_dict = {
        "acoustic": 0,
        "afrobeat": 1,
        "alt-rock": 2,
        "alternative": 3,
        "ambient": 4,
        "anime": 5,
        "black-metal": 6,
        "bluegrass": 7,
        "blues": 8,
        "brazil": 9,
        "breakbeat": 10,
        "british": 11,
        "cantopop": 12,
        "chicago-house": 13,
        "children": 14,
        "chill": 15,
        "classical": 16,
        "club": 17,
        "comedy": 18,
        "country": 19,
        "dance": 20,
        "dancehall": 21,
        "death-metal": 22,
        "deep-house": 23,
        "detroit-techno": 24,
        "disco": 25,
        "disney": 26,
        "drum-and-bass": 27,
        "dub": 28,
        "dubstep": 29,
        "edm": 30,
        "electro": 31,
        "electronic": 32,
        "emo": 33,
        "folk": 34,
        "forro": 35,
        "french": 36,
        "funk": 37,
        "garage": 38,
        "german": 39,
        "gospel": 40,
        "goth": 41,
        "grindcore": 42,
        "groove": 43,
        "grunge": 44,
        "guitar": 45,
        "happy": 46,
        "hard-rock": 47,
        "hardcore": 48,
        "hardstyle": 49,
        "heavy-metal": 50,
        "hip-hop": 51,
        "honky-tonk": 52,
        "house": 53,
        "idm": 54,
        "indian": 55,
        "indie": 56,
        "indie-pop": 57,
        "industrial": 58,
        "iranian": 59,
        "j-dance": 60,
        "j-idol": 61,
        "j-pop": 62,
        "j-rock": 63,
        "jazz": 64,
        "k-pop": 65,
        "kids": 66,
        "latin": 67,
        "latino": 68,
        "malay": 69,
        "mandopop": 70,
        "metal": 71,
        "metalcore": 72,
        "minimal-techno": 73,
        "mpb": 74,
        "new-age": 75,
        "opera": 76,
        "pagode": 77,
        "party": 78,
        "piano": 79,
        "pop": 80,
        "pop-film": 81,
        "power-pop": 82,
        "progressive-house": 83,
        "psych-rock": 84,
        "punk": 85,
        "punk-rock": 86,
        "r-n-b": 87,
        "reggae": 88,
        "reggaeton": 89,
        "rock": 90,
        "rock-n-roll": 91,
        "rockabilly": 92,
        "romance": 93,
        "sad": 94,
        "salsa": 95,
        "samba": 96,
        "sertanejo": 97,
        "show-tunes": 98,
        "singer-songwriter": 99,
        "ska": 100,
        "sleep": 101,
        "songwriter": 102,
        "soul": 103,
        "spanish": 104,
        "study": 105,
        "swedish": 106,
        "synth-pop": 107,
        "tango": 108,
        "techno": 109,
        "trance": 110,
        "trip-hop": 111,
        "turkish": 112,
        "world-music": 113
    }

    track_genre = list(track_genre_dict.keys())[list(track_genre_dict.values()).index(predictions[0])]
    print(f"Genre prédit : {track_genre}")

except Exception as e:
    print(f"Erreur : {e}")
