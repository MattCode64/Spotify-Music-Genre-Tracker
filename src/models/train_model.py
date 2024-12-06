import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Define the main training function
def train_and_log_model():
    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Define the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Start MLFlow run
    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained and logged with accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_and_log_model()
