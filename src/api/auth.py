import sqlite3
import bcrypt
import os

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel


router = APIRouter()
security = HTTPBasic()
db_path = "data/processed/database.db"

class LoginData(BaseModel):
    username: str
    password: str


def get_db_connection():
    
    if not os.path.exists(db_path):
        raise HTTPException(status_code=500, detail="Database not found.")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_user_credentials(username):
    try:
        conn = get_db_connection()
        user = conn.execute(
            "SELECT username, password FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user:
            return {"username": user["username"], "password": user["password"]}
    
    except sqlite3.DatabaseError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


def authenticate(credentials: HTTPBasicCredentials):

    username = credentials.username
    password = credentials.password
    user_data = load_user_credentials(username)

    # Vérifification des données utilisateur
    if user_data and bcrypt.checkpw(password.encode('utf-8'), user_data["password"].encode('utf-8')):
        return username
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

@router.post("/login")
def login(credentials: HTTPBasicCredentials = Depends(security)):

    user = authenticate(credentials)
    return {"message": f"Login successful for {user}"}


def test_user_in_db(username):
    conn = get_db_connection()
    user = conn.execute(
        "SELECT username, password FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    if user:
        print(f"User found: {user['username']}")
    else:
        print(f"User '{username}' not found in the database")
        


if __name__ == "__main__" :
    test_user_in_db("administrateur")
