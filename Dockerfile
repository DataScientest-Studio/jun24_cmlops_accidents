# Étape 1 : Choisir l'image de base
FROM python:3.9-slim

# Étape 2 : Définir le répertoire de travail dans le container
WORKDIR /app

# Étape 3 : Copier les fichiers requirements et installer les dépendances
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Étape 4 : Copier le reste du code de l'application dans le répertoire de travail
COPY . .

# Étape 5 : Spécifier la commande de démarrage
CMD ["python", "src/main.py"]
