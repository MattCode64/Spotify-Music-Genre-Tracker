import os

import joblib

from src.utils.dagshub_interactions import upload_file_to_dagshub


def save_model_locally(model, file_path):
    """
    Save the model_name to a local file.

    :param model: The model_name to be saved.
    :param file_path: The path where the model_name will be saved.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved locally at {file_path}")


def upload_model_to_dagshub(repo_name, bucket_name, file_path, key):
    """
    Upload the model_name file to DagsHub.

    :param repo_name: The name of the DagsHub repository.
    :param bucket_name: The name of the bucket in DagsHub.
    :param file_path: The local path of the model_name file to be uploaded.
    :param key: The key (path) in the DagsHub bucket where the file will be stored.
    """
    upload_file_to_dagshub(repo_name, bucket_name, file_path, key)
    print(f"Model uploaded to DagsHub at {key}")
