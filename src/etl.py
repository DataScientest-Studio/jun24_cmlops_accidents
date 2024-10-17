from data.make_dataset import save_final_dataset, merge_datasets
from features.build_features import transform_data, build_model_features, select_variables_and_one_hot, save_test_train
from models.train_model import load_data, train_model, save_model
from models.predict_model import evaluate_model
from config import PROCESSED_DATA_DIR
import os

def run_etl():
    """
    Exécute la pipeline ETL complète.
    """
    # S'ssurer que le dossier de sortie existe
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
    transformed_df = transform_data(df_final)
    print("Données transformées.")

    # Étape 4: Sauvegarder le DataFrame transformé dans data/processed
    transformed_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_transformed.csv')
    print("Sauvegarde du DataFrame transformé...")
    save_final_dataset(transformed_df, transformed_output_path)
    print(f"Le fichier transformé a été sauvegardé à {transformed_output_path}")

    # Étape 5: Création des features du modèle
    print("Création des features du modèle...")
    features_df = build_model_features(transformed_df)
    print("Features créées")

    # Étape 6: Sauvegarder le DataFrame avec features dans data/processed
    features_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_with_features.csv')
    print("Sauvegarde du DataFrame transformé...")
    save_final_dataset(features_df, features_output_path)
    print(f"Le fichier transformé a été sauvegardé à {features_output_path}")

    # Étape 7: Sélection des variables, OneHotEncoding et Undersampling
    print("Sélection des variables, OneHotEncoding et Undersampling...")
    encoded_df = select_variables_and_one_hot(features_df)
    print("Données encodées")

    # Étape 8: Sauvegarder les dataframes train et test encodés
    print("Sauvegarde des dataframes encodés...")
    save_test_train(encoded_df)
    print(f"Les dataframes test et train ont été sauvegardé")

    # Étape 9: Charger le df d'entrainement pour entrainer le modèle
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train_15variables_stratified.csv')
    print("Chargement des données d'entrainement...")
    train_data = load_data(train_path)
    print("Données d'entrainement chargées")

    # Étape 10: Entrainer le modèle
    print("Entrainement du modèle...")
    model = train_model(train_data)
    print("Modèle entrainé")

    # Étape 11: Sauvegarder le modèle
    print("Sauvegarde du modèle...")
    save_model(model)

    # Étape 12: Charger le df de test pour entrainer le modèle
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test_15variables_stratified.csv')
    print("Chargement des données de test...")
    test_data = load_data(test_path)
    print("Données de test chargées")

    # Étape 13: Evaluation du modèle sur les données de test
    print("Evaluation du modèle...")
    evaluate_model(test_data, model)
