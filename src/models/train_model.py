import os
import mlflow
import mlflow.sklearn
from mlflow.metrics import recall_score
from networkx.algorithms.tournament import score_sequence
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from dotenv import load_dotenv
import dagshub

load_dotenv()


def train_and_log_model(experiment_name="default"):
    # Configurer MLFlow en mode local
    DAGSHUB_URI = "https://dagshub.com/MattCode64/MelodAI.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(DAGSHUB_URI)

    # Charger le dataset
    try:
        data = pd.read_csv("/home/matthieu/UbuntuData/PycharmProjects/MelodAI/data/processed/final_dataset.csv")
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return

    X = data.drop(columns=['track_genre_encoded'])
    y = data['track_genre_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Créer ou utiliser une expérience existante
    mlflow.set_experiment(experiment_name)

    # Début d'un run
    with mlflow.start_run():
        # Définir et entraîner le modèle
        max_depth = 13
        rand_state = 42

        model = RandomForestClassifier(max_depth=max_depth, random_state=rand_state)
        model.fit(X_train, y_train)

        # Évaluer les performances
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')

        # Logger les paramètres
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", rand_state)

        # Logger les métriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

        # Enregistrer le modèle
        mlflow.sklearn.log_model(model, "model")

        # Sauvegarder les métriques dans un fichier CSV
        metrics_df = pd.DataFrame({
            "accuracy": [accuracy],
            "f1_score": [f1]
        })
        metrics_df.to_csv("/home/matthieu/UbuntuData/PycharmProjects/MelodAI/reports/training_metrics.csv", index=False)

        print(f"Modèle entraîné et loggé avec une accuracy de : {accuracy:.4f}")
        print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    dagshub.init(repo_owner='MattCode64', repo_name='MelodAI', mlflow=True)

    train_and_log_model(experiment_name="Run Test Local")
