import pandas as pd
import logging

from sklearn.metrics import classification_report
from features.build_features import select_variables_and_one_hot, save_test_train
from models.train_model import load_data, train_model, save_model
from models.predict_model import evaluate_model
from config import PROCESSED_DATA_DIR
import os

# Configuration du logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[
                        logging.FileHandler("model_pipeline.log", encoding="utf-8"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def model_pipeline():
    try:
        features_output_path = os.path.join(PROCESSED_DATA_DIR, 'df_with_features.csv')
        features_df = pd.read_csv(features_output_path)

        # Étape 7: Sélection des variables, OneHotEncoding et Undersampling
        logger.info("Sélection des variables, OneHotEncoding et Undersampling...")
        X_train, X_test, y_train, y_test = select_variables_and_one_hot(features_df)
        logger.info("Données encodées")

        # Étape 8: Sauvegarder les dataframes train et test encodés
        logger.info("Sauvegarde des dataframes encodés...")
        save_test_train(X_train, X_test, y_train, y_test)
        logger.info("Les dataframes test et train ont été sauvegardés")

        # Charger les données d'entraînement et de test
        train_path = os.path.join(PROCESSED_DATA_DIR, 'train_15variables_stratified.csv')
        test_path = os.path.join(PROCESSED_DATA_DIR, 'test_15variables_stratified.csv')

        # Étape 9: Entraînement du modèle dans une boucle
        recall_class_1, recall_class_2 = 0, 0  # Initialisation avec des scores faibles
        max_attempts = 3
        attempts = 0

        while (recall_class_1 <= 0.65 or recall_class_2 <= 0.65) and attempts < max_attempts:
            attempts += 1
            logger.info(f"Tentative {attempts}: Chargement des données d'entraînement...")
            train_X, train_y = load_data(train_path)
            logger.info("Données d'entraînement chargées")

            logger.info("Entraînement du modèle...")
            model = train_model(train_X, train_y)
            logger.info("Modèle entraîné")

            logger.info("Chargement des données de test...")
            test_X, test_y = load_data(test_path)
            logger.info("Données de test chargées")

            logger.info("Évaluation du modèle...")
            report = evaluate_model(model, test_X, test_y)

            # Calcul du rappel pour les classes 1 et 2
            recall_class_1 = report['1']['recall']
            recall_class_2 = report['2']['recall']
            logger.info(f"Rappel pour la classe 1 : {recall_class_1:.2f}")
            logger.info(f"Rappel pour la classe 2 : {recall_class_2:.2f}")

            if recall_class_1 > 0.65 and recall_class_2 > 0.65:
                logger.info("Le rappel pour les classes 1 et 2 est supérieur à 65%, sauvegarde du modèle...")
                save_model(model)
                break  # Sortie de la boucle une fois le modèle sauvegardé
            else:
                logger.warning(f"Rappel insuffisant à la tentative {attempts}, relance de l'entraînement...")

        if attempts == max_attempts:
            logger.error(f"Le modèle n'a pas atteint les performances requises après {max_attempts} tentatives.")

    except Exception as e:
        logger.error(f"Une erreur est survenue lors de l'exécution de la pipeline : {str(e)}")

    logger.info("Processus terminé.")