import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

# Import du lazy_import personnalisé
from api.lazy_imports import get_web_searcher

# Configurer le logging
logger = logging.getLogger(__name__)

class WebSearchResponse(BaseModel):
    query: str
    result: str
    sources: list[str]
    metadata: dict

web_router = APIRouter()

@web_router.get("/search", response_model=WebSearchResponse)
async def web_search(
    query: str,
    model_id: str = "gpt-4o-mini",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> WebSearchResponse:
    """
    Endpoint pour effectuer une recherche web via l'agent Web Searcher.
    """
    logger.info(f"Recherche web: query='{query}', user_id='{user_id}'")
    
    try:
        # L'import ne se fait qu'à ce moment-là
        web_searcher = get_web_searcher(
            model_id=model_id, 
            user_id=user_id, 
            session_id=session_id,
            debug_mode=False,
            stream=False
        )
        
        # Utiliser l'agent pour effectuer la recherche
        try:
            result = web_searcher.run(query, stream=False)
            
            # Extraction du contenu entre content="..."
            import re
            content_match = re.search(r'content="(.*?)"', str(result))
            if content_match:
                result = content_match.group(1)
            
            logger.info(f"Résultats de recherche obtenus : {result}")
            
        except Exception as search_error:
            logger.error(f"Erreur lors de la recherche : {search_error}")
            result = f"Erreur de recherche : {search_error}"
        
        return WebSearchResponse(
            query=query,
            result=str(result),
            sources=["llm-axe OnlineAgent"],
            metadata={
                "model_id": model_id,
                "user_id": user_id
            }
        )
    
    except Exception as e:
        logger.exception(f"Erreur globale lors de la recherche web: {e}")
        
        return WebSearchResponse(
            query=query,
            result=f"Erreur système : {e}",
            sources=[],
            metadata={
                "model_id": model_id,
                "user_id": user_id
            }
        )