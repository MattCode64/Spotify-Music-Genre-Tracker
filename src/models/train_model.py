import gc
import logging
import os

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from src.data.make_dataset import split_and_save_datasets

logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Dynamically determine the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/final_dataset.csv")
METRICS_PATH = os.path.join(PROJECT_ROOT, "reports/training_metrics.csv")


def train_and_log_model(experiment_name="default", path=None, target=None, params=dict):
    # Configure MLFlow for local tracking
    DAGSHUB_URI = "https://dagshub.com/MattCode64/MelodAI.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(DAGSHUB_URI)

    # Load the dataset
    try:
        if path:
            data = pd.read_csv(path)
        else:
            data = pd.read_csv(PROCESSED_DATA_PATH)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Split the data (train and test)
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = split_and_save_datasets(
            data.drop(columns=[target]),
            data[target],
            test_size=0.5,
            stratify=True,
            random_state=42
        )
    except Exception as e:
        logging.error(f"Error during dataset splitting: {e}")
        return

    # Create or use an existing experiment
    mlflow.set_experiment(experiment_name)

    # Start an MLFlow run
    with mlflow.start_run():
        try:
            model = RandomForestClassifier(**params)
            logging.info("Training model...")
            model.fit(X_train, y_train)
            logging.info("Model trained successfully!")

            # Evaluate performance
            predictions = model.predict(X_test)
            logging.info(f"Predictions: {predictions}")

            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average="weighted")
            precision = precision_score(y_test, predictions, average="weighted")
            recall = recall_score(y_test, predictions, average="weighted")

            if params is not None:
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            # Save metrics to a CSV file
            metrics_df = pd.DataFrame({
                "accuracy": [accuracy],
                "f1_score": [f1],
                "precision": [precision],
                "recall": [recall]
            })
            os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
            metrics_df.to_csv(METRICS_PATH, index=False)

            logging.info(f"Model trained and logged with accuracy: {accuracy:.4f}")
            logging.info(f"F1 Score: {f1:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")

        except MemoryError:
            logging.error("Memory error during model training. Consider reducing the batch size or optimizing memory usage.")
        except Exception as e:
            logging.error(f"Unexpected error during training: {e}")

        finally:
            # Free memory
            gc.collect()


if __name__ == "__main__":
    # Initialize DagsHub integration
    try:
        dagshub.init(repo_owner="MattCode64", repo_name="MelodAI", mlflow=True)
    except Exception as e:
        logging.error(f"Error initializing DagsHub: {e}")

    # Train the model and log results
    try:
        train_and_log_model(
            experiment_name="Hi khodor",
            path=PROCESSED_DATA_PATH,
            target='track_genre_encoded',
            params={}
        )
    except Exception as e:
        logging.error(f"Error during training execution: {e}")
