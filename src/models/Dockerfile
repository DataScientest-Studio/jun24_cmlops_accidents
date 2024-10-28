# Étape 1 : Choisir l'image de base
FROM python:3.9-slim

# Étape 2 : Définir le répertoire de travail
WORKDIR /app

# Étape 3 : Copier les fichiers requirements.txt et installer les dépendances
COPY src/models/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Étape 4 : Copier tout le contenu du dossier src
COPY src/models /app/models/
COPY src/models/build_features.py /app/
COPY src/models/config.py /app/
COPY src/models/model_pipeline.py /app/
COPY src/models/predict_model.py /app/
COPY src/models/train_model.py /app/

# Étape 5 : Spécifier la commande de démarrage
CMD ["python", "models/train_model.py"]