import pandas as pd
import numpy as np
from config import PROCESSED_DATA_DIR, OUTPUT_FILE_PATH

def load_final_dataset():
    """
    Charge le DataFrame final à partir du fichier CSV.
    """
    df = pd.read_csv(OUTPUT_FILE_PATH)  # Charger le df final depuis le chemin spécifié
    return df

def homogenize_hour_format(row):
    """Homogénéise le format de l'heure dans la colonne 'hrmn'."""
    hour_str = str(row['hrmn'])
    if row['an'] < 2019:
        minutes = hour_str[-2:].zfill(2)
        hour = hour_str[:-2].zfill(2)
        return f'{hour}:{minutes}'
    return hour_str

def transform_data(df):
    """
    Transforme le DataFrame selon les règles définies.
    """

    # Définir un dictionnaire de mapping pour convertir les valeurs de 'an'
    mapping = {5: 2005, 6: 2006, 7: 2007, 8: 2008, 9: 2009, 10: 2010, 
               11: 2011, 12: 2012, 13: 2013, 14: 2014, 15: 2015, 
               16: 2016, 17: 2017, 18: 2018, 2019: 2019, 2020: 2020,
               2021: 2021, 2022: 2022}

    # Appliquer la transformation en utilisant le mapping
    df['an'] = df['an'].map(mapping)

    # Homogénéiser le format de l'heure
    df['hrmn'] = df.apply(homogenize_hour_format, axis=1)

    # Filtrer les valeurs de 'an' supérieures à 2013
    df = df.loc[df['an'] > 2013]

    # Supprimer les lignes où 'grav' est égal à -1
    df = df[df['grav'] != -1]

    # Suppression des colonnes inutiles
    cols_to_drop = ['secu', 'secu1', 'secu2', 'secu3', 'motor', 'lartpc',
                    'env1', 'vma', 'occutc', 'lat', 'long', 'v1', 'v2', 
                    'pr', 'pr1', 'larrout', 'gps', 'Num_Acc', 
                    'num_veh', 'id_vehicule_y', 'id_vehicule_x', 'id_usager',
                    'adr', 'com', 'dep', 'voie', 'nbv', 'actp', 
                    'etatp', 'locp', 'obs', 'senc', 'trajet', 'vosp']
    df.drop(columns=cols_to_drop, inplace=True)

    df = map_categories(df)

    # Remplacer les valeurs manquantes par le mode le plus fréquent
    for column in ['manv', 'obsm', 'place', 'int', 'situ', 'choc', 'atm', 'catr', 'surf']:
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)

    # Supprimer les doublons et réinitialiser l'index
    df = df.drop_duplicates().reset_index(drop=True)

    return df

def map_categories(df):
    """Mapper les nouvelles valeurs pour les colonnes catégorielles."""

    # Variable 'manv'
    manv_mapping = {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 2,
        12: 2,
        13: 3,
        14: 3,
        15: 4,
        16: 4,
        17: 5,
        18: 5,
        19: 6,
        20: 6,
        21: 6,
        22: 6,
        23: 6,
        24: 6,
        25: 6,
        26: 6
    }
    df['manv'] = df['manv'].map(manv_mapping).fillna(np.nan)

    # Variable 'obsm'
    obsm_mapping = {
        0: 1,
        1: 2,
        2: 3,
        4: 4,
        5: 4,
        6: 4,
        9: 4
    }
    df['obsm'] = df['obsm'].map(obsm_mapping).fillna(np.nan)

    # Variable 'place'
    place_mapping = {
        1: 1,
        2: 2,
        3: 2,
        4: 3,
        5: 3,
        6: 2,
        7: 3,
        8: 3,
        9: 3,
        10: 4
    }
    df['place'] = df['place'].map(place_mapping).fillna(np.nan)

    # Variable 'int'
    int_mapping = {
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 3,
        7: 4,
        8: 2,
        9: 2,
    }
    df['int'] = df['int'].map(int_mapping).fillna(np.nan)

    # Variable 'situ'
    situ_mapping = {
        1: 1,
        2: 3,
        3: 2,
        4: 3,
        5: 3,
        6: 3,
        8: 3,
    }
    df['situ'] = df['situ'].map(situ_mapping).fillna(np.nan)

    # Variable 'choc'
    choc_mapping = {
        0: 1,
        1: 2,
        2: 2,
        3: 2,
        4: 3,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        9: 4,
    }
    df['choc'] = df['choc'].map(choc_mapping).fillna(np.nan)

    # Variable 'atm'
    atm_mapping = {
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 3,
        6: 4,
        7: 3,
        8: 1,
        9: 1,
    }
    df['atm'] = df['atm'].map(atm_mapping).fillna(np.nan)

    # Variable 'catr'
    catr_mapping = {
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 5,
        6: 5,
        7: 5,
        9: 5,
    }
    df['catr'] = df['catr'].map(catr_mapping).fillna(np.nan)

    # Variable 'surf'
    surf_mapping = {
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 3,
    }
    df['surf'] = df['surf'].map(surf_mapping).fillna(np.nan)

    return df


def save_transformed_dataset(df, output_path):
    """
    Sauvegarde le DataFrame transformé dans le chemin spécifié.
    """
    df.to_csv(output_path, index=False)
    print(f"Le fichier transformé a été sauvegardé à {output_path}")

def main():
    # Chargement du DataFrame final
    df_final = load_final_dataset()

    # Transformation des données
    transformed_df = transform_data(df_final)

    # Sauvegarde du DataFrame transformé
    save_transformed_dataset(transformed_df, "data/processed/transformed_dataset.csv")