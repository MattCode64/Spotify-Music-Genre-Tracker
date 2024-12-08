import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from dotenv import load_dotenv
import dagshub

# Load environment variables
load_dotenv()

# Dynamically determine the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/final_dataset.csv")
METRICS_PATH = os.path.join(PROJECT_ROOT, "reports/training_metrics.csv")


def train_and_log_model(experiment_name="default"):
    # Configure MLFlow for local tracking
    DAGSHUB_URI = "https://dagshub.com/MattCode64/MelodAI.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(DAGSHUB_URI)

    # Load the dataset
    try:
        data = pd.read_csv(PROCESSED_DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    X = data.drop(columns=["track_genre_encoded"])
    y = data["track_genre_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Create or use an existing experiment
    mlflow.set_experiment(experiment_name)

    # Start an MLFlow run
    with mlflow.start_run():
        # Define and train the model
        max_depth = 13
        rand_state = 42

        model = RandomForestClassifier(max_depth=max_depth, random_state=rand_state)
        model.fit(X_train, y_train)

        # Evaluate performance
        predictions = model.predict(X_test)
        print(f"Predictions: {predictions}")
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        # Log parameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", rand_state)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

        # Save the model
        input_example = X_test.iloc[:5]
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        # Save metrics to a CSV file
        metrics_df = pd.DataFrame({
            "accuracy": [accuracy],
            "f1_score": [f1]
        })
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        metrics_df.to_csv(METRICS_PATH, index=False)

        print(f"Model trained and logged with accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    # Initialize DagsHub integration
    dagshub.init(repo_owner="MattCode64", repo_name="MelodAI", mlflow=True)

    # Train the model and log results
    train_and_log_model(experiment_name="Run Test Local")
