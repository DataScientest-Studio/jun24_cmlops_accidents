from data.make_dataset import save_final_dataset, merge_datasets
from features.build_features import transform_data, build_model_features, select_variables_and_one_hot, save_test_train
from models.train_model import load_data, train_model, save_model
from models.predict_model import evaluate_model
from config import PROCESSED_DATA_DIR

import os
import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("etl_pipeline.log"),  # Sauvegarde dans root, modifiable au besoin
        logging.StreamHandler()  # Affichage dans la console
    ]
)
logger = logging.getLogger(__name__)

def run_etl():
    """
    Exécute la pipeline ETL complète.
    """
    # S'assurer que le dossier de sortie existe
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Étape 1: Extraire et fusionner les données
    logger.info("Extraction et fusion des données...")
    df_final = merge_datasets()
    logger.info("Données extraites et fusionnées.")
    
    # Étape 2: Sauvegarder le DataFrame final dans data/processed
    final_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_final.csv')
    logger.info("Sauvegarde du DataFrame concaténé et homogénéisé...")
    save_final_dataset(df_final, final_output_path)
    logger.info(f"Le fichier concaténé et homogénéisé a été sauvegardé à {final_output_path}")

    # Étape 3: Transformation des données
    logger.info("Transformation des données...")
    transformed_df = transform_data(df_final)
    logger.info("Données transformées.")

    # Étape 4: Sauvegarder le DataFrame transformé dans data/processed
    transformed_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_transformed.csv')
    logger.info("Sauvegarde du DataFrame transformé...")
    save_final_dataset(transformed_df, transformed_output_path)
    logger.info(f"Le fichier transformé a été sauvegardé à {transformed_output_path}")

    # Étape 5: Création des features du modèle
    logger.info("Création des features du modèle...")
    features_df = build_model_features(transformed_df)
    logger.info("Features créées")

    # Étape 6: Sauvegarder le DataFrame avec features dans data/processed
    features_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_with_features.csv')
    logger.info("Sauvegarde du DataFrame transformé...")
    save_final_dataset(features_df, features_output_path)
    logger.info(f"Le fichier transformé a été sauvegardé à {features_output_path}")