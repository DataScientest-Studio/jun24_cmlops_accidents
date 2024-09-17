import logging
from logging.handlers import RotatingFileHandler
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from src.api.auth import authenticate


security = HTTPBasic()

router = APIRouter()

def configure_logging():
    log_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=1 * 1024 * 1024,
        backupCount=3,
    )
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


configure_logging()


@router.get("/")
def get_logs(credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)

    logging.info("Logs endpoint accessed")
    try:
        with open("logs/app.log", "r") as f:
            logs = f.read()
        return {"logs": logs}
    except Exception as e:
        logging.error(f"Failed to read logs: {str(e)}")
        return {"error": "Failed to read logs"}
