from fastapi import APIRouter
import logging

# Configuration du logger
logger = logging.getLogger(__name__)

# Importer les routes des différents agents
from .web_router import web_router    
from .orchestrator_router import orchestrator_router
from .user_proxy_router import user_proxy_router
    
# Créer le routeur principal V1
v1_router = APIRouter(prefix="/v1")

# Inclure les routes spécifiques à chaque agent
v1_router.include_router(web_router, prefix="/web", tags=["Web Search"])
v1_router.include_router(orchestrator_router, prefix="/router", tags=["Orchestrator"])
v1_router.include_router(user_proxy_router, prefix="/user_proxy", tags=["User Proxy"])

logger.info("Routes V1 initialisées avec succès")