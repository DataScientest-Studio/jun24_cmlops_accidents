from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import logging
from auth import authenticate  


router = APIRouter()
security = HTTPBasic()

# Route pour récupérer les logs sécurisée avec HTTPBasic
@router.get("/")
def get_logs(credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)

    logging.info("Logs endpoint accessed")
    try:
        with open("../../logs/app.log", "r") as f:
            logs = f.read()
        return {"logs": logs}
    except Exception as e:
        logging.error(f"Failed to read logs: {str(e)}")
        return {"error": "Failed to read logs"}
