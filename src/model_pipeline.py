import pandas as pd

from features.build_features import select_variables_and_one_hot, save_test_train
from models.train_model import load_data, train_model, save_model
from models.predict_model import evaluate_model
from config import PROCESSED_DATA_DIR
import os

def model_pipeline():

    features_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_with_features.csv')
    features_df = pd.read_csv(features_output_path)

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