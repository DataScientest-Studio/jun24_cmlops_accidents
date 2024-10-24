
import pandas as pd
from config import RAW_DATA_DIR
import os

def load_data_by_year(file_prefix, start_year, end_year, sep_default=',', sep_special=None):
    """
    Charge et concatène des fichiers CSV d'une période donnée en ajustant les séparateurs de colonnes si nécessaire.
    
    :param file_prefix: Préfixe du nom de fichier, par exemple "caracteristiques", "lieux", etc.
    :param start_year: Année de début
    :param end_year: Année de fin
    :param sep_default: Séparateur par défaut (',')
    :param sep_special: Dictionnaire indiquant l'année et le séparateur spécial (ex: {2009: '\t'})
    :return: DataFrame concaténé pour les années spécifiées
    """
    df_list = []
    for year in range(start_year, end_year + 1):
        file_name = f"{file_prefix}_{year}.csv"
        file_path = os.path.join(RAW_DATA_DIR, file_name)  # On utilise RAW_DATA_DIR qui pointe vers data/raw à la racine
        
        # Utiliser le séparateur spécial si défini pour une année spécifique, sinon utiliser le séparateur par défaut
        separator = sep_special.get(year, sep_default) if sep_special else sep_default
        
        df = pd.read_csv(file_path, sep=separator, low_memory=False, encoding='latin-1')
        df_list.append(df)
    
    return pd.concat(df_list, ignore_index=True)

def load_caracteristiques():
    """
    Charge et concatène les fichiers 'caracteristiques' de 2005 à 2022.
    """
    df_2005_2018 = load_data_by_year("caracteristiques", 2005, 2018, sep_special={2009: '\t'})
    df_2019_2022 = load_data_by_year("caracteristiques", 2019, 2022, sep_default=';')
    
    return pd.concat([df_2005_2018, df_2019_2022], ignore_index=True)

def load_lieux():
    """
    Charge et concatène les fichiers 'lieux' de 2005 à 2022.
    """
    df_2005_2018 = load_data_by_year("lieux", 2005, 2018)
    df_2019_2022 = load_data_by_year("lieux", 2019, 2022, sep_default=';')
    
    return pd.concat([df_2005_2018, df_2019_2022], ignore_index=True)

def load_usagers():
    """
    Charge et concatène les fichiers 'usagers' de 2005 à 2022.
    """
    df_2005_2018 = load_data_by_year("usagers", 2005, 2018)
    df_2019_2022 = load_data_by_year("usagers", 2019, 2022, sep_default=';')
    
    return pd.concat([df_2005_2018, df_2019_2022], ignore_index=True)

def load_vehicules():
    """
    Charge et concatène les fichiers 'vehicules' de 2005 à 2022.
    """
    df_2005_2018 = load_data_by_year("vehicules", 2005, 2018)
    df_2019_2022 = load_data_by_year("vehicules", 2019, 2022, sep_default=';')
    
    return pd.concat([df_2005_2018, df_2019_2022], ignore_index=True)

def merge_datasets():
    """
    Fusionne les différentes tables pour créer un DataFrame final.
    """
    df_usagers = load_usagers()
    df_vehicules = load_vehicules()
    df_caracteristiques = load_caracteristiques()
    df_lieux = load_lieux()

    # Fusion des tables usagers et véhicules sur 'Num_Acc' et 'num_veh'
    df_UV = pd.merge(df_usagers, df_vehicules, on=['Num_Acc', 'num_veh'])
    
    # Fusion avec les caractéristiques sur 'Num_Acc'
    df_UVC = pd.merge(df_UV, df_caracteristiques, on=["Num_Acc"])
    
    # Fusion finale avec les lieux sur 'Num_Acc'
    df_UVCL = pd.merge(df_UVC, df_lieux, on=["Num_Acc"])

    return df_UVCL

def save_final_dataset(df, output_path):
    """
    Sauvegarde le DataFrame final au chemin spécifié.
    
    :param df: DataFrame à sauvegarder
    :param output_path: Chemin du fichier de sortie
    """
    df.to_csv(output_path, index=False)