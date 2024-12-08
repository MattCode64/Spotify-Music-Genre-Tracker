import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
        data = pd.read_csv('../../data/processed/dataset_ready.csv')
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
        model = RandomForestClassifier(max_depth=89, random_state=42)
        model.fit(X_train, y_train)

        # Évaluer les performances
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Logger les paramètres
        mlflow.log_param("n_estimators", 11)
        mlflow.log_param("random_state", 42)

        # Logger les métriques
        mlflow.log_metric("accuracy", accuracy)

        # Enregistrer le modèle
        mlflow.sklearn.log_model(model, "model")

        print(f"Modèle entraîné et loggé avec une accuracy de : {accuracy:.4f}")


if __name__ == "__main__":
    dagshub.init(repo_owner='MattCode64', repo_name='MelodAI', mlflow=True)

    train_and_log_model(experiment_name="Run Test Local")
