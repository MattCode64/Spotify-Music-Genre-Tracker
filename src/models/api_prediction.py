import requests

# URL de l'API MLFlow
url = "http://127.0.0.1:5001/invocations"

# Données pour la prédiction (format dataframe_split)
payload = {
    "dataframe_split": {
        "columns": ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
                    'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'],
        "data": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]
    }
}

# Envoyer la requête POST
response = requests.post(url, json=payload)

# Afficher les résultats
if response.status_code == 200:
    print("Prédictions :", response.json())
else:
    print("Erreur :", response.text)
