import pandas as pd
import numpy as np
import os
import joblib

from config import PROCESSED_DATA_DIR

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from config import MODEL_DIR

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
    df = df.drop(columns=cols_to_drop)

    df = map_categories(df)

    # Remplacer les valeurs manquantes par le mode le plus fréquent
    for column in ['manv', 'obsm', 'place', 'int', 'situ', 'choc', 'atm', 'catr', 'surf']:
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)

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

def build_model_features(df):
    """
    Construit les catégories d'age, d'heures et de jour de la semaine, puis les encode.
    """

    # 1. Transformation de la variable horaire 'hrmn'
    # Création de la variable 'hour_cat' en fonction des heures
    df['hour_cat'] = pd.to_datetime(df['hrmn'], format='%H:%M').dt.hour

    # Réduire les catégories horaires en 3 groupes
    def reduce_categories_hour(x):
        if x in range(9, 17):
            return 1  # Heure de bureau
        elif (x in range(6, 9)) or (x in range(16, 20)):
            return 2  # Heure de pointe
        elif (x in range(20, 24)) or (x in range(0, 6)):
            return 3  # Nuit
        else:
            return -1

    df['hour_cat'] = df['hour_cat'].apply(reduce_categories_hour)

    # 2. Imputation des valeurs manquantes (-1) par le mode le plus fréquent de chaque colonne
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df.replace(-1, np.nan)), columns=df.columns)

    # 3. Calcul de l'âge de l'usager au moment de l'accident
    df['age_usag'] = df['an'] - df['an_nais']

    # 4. Catégorisation de l'âge à l'aide d'un transformateur personnalisé
    class AgeCat(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return pd.DataFrame(pd.cut(X['age_usag'], 
                                       bins=[0, 12, 18, 30, 58, np.max(X['age_usag'])],
                                       labels=['Enfant', 'Adolescent', 'Jeune adulte', 'Adulte', 'Senior']))

    # Appliquer le transformateur personnalisé pour catégoriser l'âge
    age_categorized = AgeCat()
    df['age_category'] = age_categorized.fit_transform(df[['age_usag']])

    # 5. Ajouter une colonne pour le jour de la semaine
    def jour_de_la_semaine(date):
        jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        return jours[date.weekday()]

    # Convertir la colonne 'hrmn' pour s'assurer que toutes les heures sont au format HH:MM
    df["hrmn"] = df["hrmn"].apply(lambda x: str(x).zfill(4))

    # Création d'une colonne datetime
    df["datetime_str"] = (
        df["an"].astype(str) + "-" + df["mois"].astype(str) + "-" + df["jour"].astype(str) + " " + df["hrmn"]
    )
    df["datetime"] = pd.to_datetime(df["datetime_str"], format="%Y-%m-%d %H:%M")
    df["jour_sem"] = df["datetime"].apply(jour_de_la_semaine)

    # 6. Encodage des variables catégorielles avec LabelEncoder
    # Encodage de la catégorie d'âge
    label_encoder_age_category = LabelEncoder()
    df['age_category_encoded'] = label_encoder_age_category.fit_transform(df['age_category'])

    # Encodage du jour de la semaine
    label_encoder_jour_sem = LabelEncoder()
    df['jour_sem_encoded'] = label_encoder_jour_sem.fit_transform(df['jour_sem'])

    # 7. Nettoyage des colonnes inutiles
    # Suppression des variables "temps" et autres colonnes utilisées temporairement
    df = df.drop(columns=['an', 'an_nais', 'jour', 'mois', 'hrmn', 'age_usag', 
                     'datetime_str', 'datetime', 'age_category', 'jour_sem'])

    # 8. Assurez-vous que toutes les colonnes 'object' sont converties en 'int'
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('int')

    # 9. Suppression des doublons
    df = df.drop_duplicates()

    return df

def select_variables_and_one_hot(df):
    """
    Sélection des variables et OneHot Encoding. On réalise également un undersampling car les données sont déséquilibrées.
    """
    y = df['grav']  # Cible

    # Liste des variables à conserver = 15 variables - Pourrait être automatisé avec un SelectKBest éventuellement
    variables_a_garder = ['catu', 'catv', 'obsm', 'place', 'manv', 'situ', 'agg', 'plan', 'age_category_encoded', 'int', 'sexe', 'lum', 'hour_cat', 'catr', 'choc']
    # Garder seulement les colonnes spécifiées
    df_15 = df.filter(variables_a_garder)
    df_15 = df_15.astype('object')

    # Utiliser un OneHotEncoder pour pouvoir le sauvegarder et le réutiliser dans l'API
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_encoded = encoder.fit_transform(df_15)

    # Sauvegarde de l'encoder pour une utilisation future
    encoder_path = os.path.join(MODEL_DIR, 'SGD_encoder.joblib')
    joblib.dump(encoder, encoder_path)

    # On retransforme X en datafrmae panda
    X = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(df_15.columns))

    # Les classes sont déséquilibrées, ont réalise donc un undersampling
    # Appliquer l'undersampling sur X et y en spécifiant un random_state
    undersample = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersample.fit_resample(X, y)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    return X_train, X_test, y_train, y_test

def save_test_train(X_train, X_test, y_train, y_test):
    """
    Concatène les dataframes de train et test obtenus afin de les sauvegarder dans data/processed
    """
    datasets = {'train': (X_train, y_train), 'test': (X_test, y_test)}

    for dataset_name, (X, y) in datasets.items():
        # Convertir X et y en DataFrames pandas
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)

        # Concaténer X et y en un seul DataFrame
        dataset_df = pd.concat([X_df, y_df], axis=1)
        
        # Exporter le DataFrame au format CSV
        output_filename = os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}_15variables_stratified.csv")
        dataset_df.to_csv(output_filename, index=False)