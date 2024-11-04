import os

# Chemin de la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Répertoires des données brutes et traitées
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')  # data/raw à la racine
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')  # data/processed à la racine

# Chemin du fichier final exporté
OUTPUT_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, 'df_final.csv')

# Répertoire des modèles
MODEL_DIR = os.path.join(BASE_DIR,'models')