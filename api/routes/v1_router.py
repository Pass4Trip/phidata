from fastapi import APIRouter
import logging

# Importer les routes existantes
from .user_proxy_router import user_proxy_router
from .web_router import web_router    
from .orchestrator_router import orchestrator_router
from .websocket_router import websocket_router  # Correction de l'import

# Configuration du logger
logger = logging.getLogger(__name__)

# Créer le routeur principal V1
v1_router = APIRouter(prefix="/v1")

# Inclure les routes
v1_router.include_router(web_router, prefix="/web", tags=["Web Search"])
v1_router.include_router(orchestrator_router, prefix="/router", tags=["Orchestrator"])
v1_router.include_router(user_proxy_router, prefix="/user_proxy", tags=["User Proxy"])
v1_router.include_router(
    websocket_router,
    prefix="/ws",
    tags=["WebSocket"]
)  # Ajout du routeur WebSocket

logger.info("Routes V1 initialisées avec succès")