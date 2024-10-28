from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle sur les données de test et affiche le rapport de classification."""
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    print("Rapport de classification sur les données de test :\n", class_report)
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)
    return class_report_dict