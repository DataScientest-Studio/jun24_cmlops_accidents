import pandas as pd
import os
import joblib

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, recall_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from models.config import MODEL_DIR

def load_data(file_path):
    """Charge les données à partir d'un fichier CSV."""
    df = pd.read_csv(file_path)
    X = df.drop('grav', axis=1)
    y = df['grav']
    return X, y

def train_model(X_train, y_train):
    """Entraîne le modèle à l'aide de GridSearchCV."""
    # Définir le scorer personnalisé pour le rappel macro sur la classe 2
    scorer = make_scorer(recall_score, average='macro', labels=[2])

    # Appliquer le sous-échantillonnage
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

    # Définir les hyperparamètres du SGDClassifier
    param_grid_sgd = {
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__loss': ['hinge', 'modified_huber'],
        'classifier__penalty': ['l2', 'l1', 'elasticnet'],
        'classifier__max_iter': [1000, 2000, 3000]
    }

    # Initialiser le classificateur SGDClassifier
    sgd = SGDClassifier(random_state=42)

    # Créer le pipeline avec normalisation et classificateur
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', sgd)
    ])

    # Initialiser GridSearchCV avec le pipeline et les hyperparamètres
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid_sgd, cv=5, scoring=scorer, n_jobs=4)

    # Exécuter la recherche sur la grille sur les données d'entraînement
    grid_search.fit(X_resampled, y_resampled)

    # Obtenir le meilleur modèle
    best_model = grid_search.best_estimator_
    print("Meilleurs paramètres:", grid_search.best_params_)

    return best_model

def save_model(model):
    """Sauvegarde le modèle entraîné."""
    model_path = os.path.join(MODEL_DIR, 'best_SGDClass.joblib')
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé à {model_path}")