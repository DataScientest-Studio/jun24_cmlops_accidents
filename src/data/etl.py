from make_dataset import save_final_dataset, merge_datasets
from build_features import transform_data, build_model_features
from config import PROCESSED_DATA_DIR

import os
import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("etl_pipeline.log", encoding="utf-8"),  # Sauvegarde dans root, modifiable au besoin
        logging.StreamHandler()  # Affichage dans la console
    ]
)
logger = logging.getLogger(__name__)

class ColumnValidationError(Exception):
    """Exception levée lorsque les colonnes d'un DataFrame ne correspondent pas aux attentes."""
    pass

def validate_column_count(df, expected_count, df_name):
    """
    Vérifie que le DataFrame a le bon nombre de colonnes.

    :param df: Le DataFrame à valider
    :param expected_count: Nombre de colonnes attendues
    :param df_name: Nom du DataFrame (pour les logs et exceptions)
    :raises ColumnValidationError: Si le nombre de colonnes n'est pas conforme
    """
    actual_count = len(df.columns)
    
    if actual_count != expected_count:
        raise ColumnValidationError(f"{df_name} a {actual_count} colonnes au lieu des {expected_count} attendues.")
    
    logger.info(f"{df_name} a le bon nombre de colonnes ({actual_count}).")

def validate_column_names(df, expected_columns, df_name):
    """
    Vérifie que le DataFrame a les bons noms de colonnes.

    :param df: Le DataFrame à valider
    :param expected_columns: Liste des noms de colonnes attendues
    :param df_name: Nom du DataFrame (pour les logs et exceptions)
    :raises ColumnValidationError: Si les noms de colonnes ne sont pas conformes
    """
    actual_columns = list(df.columns)
    
    if actual_columns != expected_columns:
        raise ColumnValidationError(f"{df_name} a des colonnes inattendues. Colonnes actuelles : {actual_columns}. Colonnes attendues : {expected_columns}.")
    
    logger.info(f"Les colonnes de {df_name} sont correctement nommées pour les étapes d'entrainement des modèles.")

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

    # Validation du DataFrame df_final
    df_final_expected_columns = 59
    validate_column_count(df_final, df_final_expected_columns, "df_final") # Nous vérifions seulement le nombre de colonnes, pas les noms

    # Étape 2: Sauvegarder le DataFrame final dans data/processed
    final_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_final.csv')
    logger.info("Sauvegarde du DataFrame concaténé et homogénéisé...")
    save_final_dataset(df_final, final_output_path)
    logger.info(f"Le fichier concaténé et homogénéisé a été sauvegardé à {final_output_path}")

    # Étape 3: Transformation des données
    logger.info("Transformation des données...")
    transformed_df = transform_data(df_final)
    logger.info("Données transformées.")

    # Validation du DataFrame transformed_df
    df_transformed_expected_columns = 25
    validate_column_count(transformed_df, df_transformed_expected_columns, "df_transformed") # Seulement le nombre de colonnes toujours

    # Étape 4: Sauvegarder le DataFrame transformé dans data/processed
    transformed_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_transformed.csv')
    logger.info("Sauvegarde du DataFrame transformé...")
    save_final_dataset(transformed_df, transformed_output_path)
    logger.info(f"Le fichier transformé a été sauvegardé à {transformed_output_path}")

    # Étape 5: Création des features du modèle
    logger.info("Création des features du modèle...")
    features_df = build_model_features(transformed_df)
    logger.info("Features créées")

    # Validation du DataFrame features_df
    df_with_features_expected_columns = [
        "place", "catu", "grav", "sexe", "catv", "obsm", "choc", "manv", "lum", "agg", "inter", "atm", "col", "catr", "circ", 
        "prof", "plan", "surf", "infra", "situ", "hour_cat", "age_category_encoded", "jour_sem_encoded"
    ]
    df_with_features_expected_count = len(df_with_features_expected_columns)
    # Vérification du nombre et des noms de colonnes pour df_with_features
    validate_column_count(features_df, df_with_features_expected_count, "df_with_features")
    validate_column_names(features_df, df_with_features_expected_columns, "df_with_features")

    # Étape 6: Sauvegarder le DataFrame avec features dans data/processed
    features_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_with_features.csv')
    logger.info("Sauvegarde du DataFrame transformé...")
    save_final_dataset(features_df, features_output_path)
    logger.info(f"Le fichier transformé a été sauvegardé à {features_output_path}")

if __name__ == "__main__":
    run_etl()