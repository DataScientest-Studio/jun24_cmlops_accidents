from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'pipeline_dag',
    default_args=default_args,
    description='Pipeline complet de traitement et entraînement',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 11, 1),
    catchup=False,
) as dag:

    # Étape 1 : Exécuter le traitement des données dans le conteneur data
    data_processing = BashOperator(
        task_id='data_processing',
        bash_command='kubectl exec deployment/data-deployment -- python /app/data/etl.py'
    )

    # Étape 2 : Entraîner le modèle dans le conteneur models
    model_training = BashOperator(
        task_id='model_training',
        bash_command='kubectl exec deployment/models-deployment -- python /app/models/train_model.py'
    )

    # Étape 3 : Lancer les prédictions dans le conteneur API
    api_prediction = BashOperator(
        task_id='api_prediction',
        bash_command='kubectl exec deployment/api-deployment -- python /app/api/predict.py'
    )

    # Définir l'ordre d'exécution
    data_processing >> model_training >> api_prediction
