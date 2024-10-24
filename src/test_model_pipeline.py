from model_pipeline import model_pipeline

if __name__ == "__main__":
    print("Lancement de la pipeline du modèle...")
    try:
        model_pipeline()
        print("La pipeline d'entrainement du modèle a été exécutée avec succès.")
    except Exception as e:
        # Capture toutes les exceptions et affiche un message d'erreur
        print(f"Une erreur est survenue lors de l'exécution de la pipeline : {e}")