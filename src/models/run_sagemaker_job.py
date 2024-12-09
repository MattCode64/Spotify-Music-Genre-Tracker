"""
Script exécuté localement (ex: via PyCharm) pour lancer un job SageMaker
à partir du script 'train_model.py'.
"""
import os

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# Récupération du rôle d'exécution SageMaker (vous pouvez le récupérer depuis la console AWS IAM)
# « Vous pouvez aussi utiliser sagemaker.get_execution_role() si vous êtes sur une instance Sagemaker
role = "arn:aws:iam::476114139467:role/service-role/AmazonSageMaker-ExecutionRole-20241209T172604"

session = sagemaker.Session(boto_session=boto3.Session())

# # Charger le dataset sur S3 (adaptez le chemin)
# train_input = session.upload_data(
#     path='data/processed/final_dataset.csv',  # le chemin local vers le dataset
#     bucket='s3://s3-melodai-project/processed/',
#     key_prefix='data/processed'
# )

# Download le dataset depuis S3 (car dataset déjà sur S3)
train_input = 's3://s3-melodai-project/processed/83/ab60d273bba7b233b418a0942de534'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAIN_MODEL_FILE = os.path.join(PROJECT_ROOT, "src/models/train_model.py")

# Définir l'estimateur scikit-learn
estimator = SKLearn(
    entry_point=TRAIN_MODEL_FILE,
    role=role,
    instance_count=1,
    instance_type='ml.t2.medium',
    framework_version='1.0-1',
    py_version='py3',
    sagemaker_session=session,
    hyperparameters={},
    dependencies=['dvc.yaml', '.dvc', 'requirements.txt']  # si besoin
)

# Lancer le job d'entraînement SageMaker
estimator.fit({'train': train_input})
