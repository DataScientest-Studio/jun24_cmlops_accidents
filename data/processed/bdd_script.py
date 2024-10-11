import bcrypt
import sqlite3

def add_user_to_db(username, password):
    # Connexion à la base de données
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # Hachage du mot de passe
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    try:
        # Insertion de l'utilisateur dans la base de données
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password),
        )
        conn.commit()
        print(f"User '{username}' added successfully.")
    except sqlite3.IntegrityError:
        print(f"User '{username}' already exists.")
    finally:
        conn.close()


# Ajouter l'utilisateur "administrateur" avec le mot de passe "easyMdp"
add_user_to_db("administrateur", "easyMdp")
