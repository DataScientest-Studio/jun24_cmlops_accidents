from data.make_dataset import save_final_dataset, merge_datasets
from features.build_features import load_final_dataset, transform_data
from config import OUTPUT_FILE_PATH, PROCESSED_DATA_DIR
import os

def run_etl():
    """
    Exécute la pipeline ETL complète.
    """
    # Assurez-vous que le dossier de sortie existe
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Étape 1: Extraire et fusionner les données
    print("Extraction et fusion des données...")
    df_final = merge_datasets()
    print("Données extraites et fusionnées.")
    
    # Étape 2: Sauvegarder le DataFrame final dans data/processed
    final_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_final.csv')
    print("Sauvegarde du DataFrame concaténé et homogénéisé...")
    save_final_dataset(df_final, final_output_path)
    print(f"Le fichier concaténé et homogénéisé a été sauvegardé à {final_output_path}")

    # Étape 3: Transformation des données
    print("Transformation des données...")
    transformed_df = transform_data(df_final)  # Appel à la fonction de transformation
    print("Données transformées.")

    # Étape 4: Sauvegarder le DataFrame transformé dans data/processed
    transformed_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_transformed.csv')
    print("Sauvegarde du DataFrame transformé...")
    save_final_dataset(transformed_df, transformed_output_path)
    print(f"Le fichier transformé a été sauvegardé à {transformed_output_path}")

