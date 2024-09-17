from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import json
from pydantic import BaseModel
import secrets

router = APIRouter()
security = HTTPBasic()


class LoginData(BaseModel):
    username: str
    password: str

def load_user_credentials():
    with open("../../data/raw/login_data.json", "r") as f:  
        return json.load(f)

def authenticate(credentials: HTTPBasicCredentials):
    
    credentials_data = load_user_credentials()
    username = credentials.username
    password = credentials.password

    # Vérif ids
    if username in credentials_data and secrets.compare_digest(
        credentials_data[username], password
    ):
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
