from fastapi import FastAPI
import logging
from dotenv import load_dotenv
from utils.logging import configure_logging
from api.routes.web__router import web_router

# Charger les variables d'environnement
load_dotenv()

# Configurer le logging
configure_logging()

app = FastAPI()

# Inclusion du routeur web
app.include_router(web_router, prefix="/web", tags=["web"])

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
