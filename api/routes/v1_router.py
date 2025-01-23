from fastapi import APIRouter
import logging
import sys
import traceback

# Configuration du logger
logger = logging.getLogger(__name__)

# Importer les routes des différents agents
try:
    from api.routes.web import web_router
    from api.routes.api_knowledge import api_knowledge_router
    from api.routes.data_analysis import data_analysis_router
    from api.routes.orchestrator_router import orchestrator_router
except ImportError as e:
    logger.error(f"Erreur d'importation des routes : {e}")
    logger.error(traceback.format_exc())
    
    # Routes de secours si l'import échoue
    web_router = APIRouter()
    api_knowledge_router = APIRouter()
    data_analysis_router = APIRouter()
    orchestrator_router = APIRouter()

# Créer le routeur principal V1
v1_router = APIRouter(prefix="/v1")

# Inclure les routes spécifiques à chaque agent
v1_router.include_router(web_router, prefix="/web", tags=["Web Search"])
v1_router.include_router(api_knowledge_router, prefix="/knowledge", tags=["API Knowledge"])
v1_router.include_router(data_analysis_router, prefix="/data", tags=["Data Analysis"])
v1_router.include_router(orchestrator_router, prefix="/router", tags=["Orchestrator"])

logger.info("Routes V1 initialisées avec succès")