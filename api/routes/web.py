from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, Dict, Any
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_429_TOO_MANY_REQUESTS
from pydantic import BaseModel, Field
import logging
import json
import re

from agents.web import get_web_searcher

logger = logging.getLogger(__name__)
web_router = APIRouter()

def clean_response(result_str: str) -> str:
    """
    Nettoie la réponse pour la rendre plus lisible
    
    Args:
        result_str (str): Chaîne de résultat brute
    
    Returns:
        str: Chaîne de résultat nettoyée
    """
    # Supprimer les délimiteurs LaTeX
    result_str = re.sub(r'\\[()[\]]', '', result_str)
    
    # Supprimer les espaces en trop
    result_str = re.sub(r'\s+', ' ', result_str).strip()
    
    return result_str

class WebSearchResponse(BaseModel):
    """
    Modèle structuré pour la réponse de recherche web
    """
    query: str = Field(..., description="La requête originale de l'utilisateur")
    result: str = Field(..., description="Résultat de la recherche ou de l'analyse")
    sources: Optional[list[str]] = Field(None, description="Sources utilisées pour générer la réponse")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées supplémentaires")
    error: Optional[str] = Field(None, description="Message d'erreur si la recherche a échoué")

@web_router.get("/search")
async def web_search(
    query: str,
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    max_retries: int = 2
) -> str:
    """
    Endpoint pour effectuer une recherche web via l'agent Web Searcher.
    
    Args:
        query (str): La requête de recherche
        model_id (Optional[str]): ID du modèle à utiliser
        user_id (Optional[str]): ID de l'utilisateur
        session_id (Optional[str]): ID de session
        max_retries (int): Nombre maximum de tentatives en cas d'échec
    
    Returns:
        str: Résultat de la recherche web
    """
    for attempt in range(max_retries + 1):
        try:
            web_searcher = get_web_searcher(
                model_id=model_id, 
                user_id=user_id, 
                session_id=session_id,
                debug_mode=True  # Activer le mode debug pour plus d'informations
            )
            
            # Exécuter la recherche
            result = web_searcher.run(query)
            
            # Convertir le résultat en chaîne de caractères
            if hasattr(result, 'content'):
                result_str = str(result.content)
            elif isinstance(result, dict):
                result_str = json.dumps(result)
            else:
                result_str = str(result)
            
            # Nettoyer la réponse
            #result_str = clean_response(result_str)
            
            return result_str
        
        except Exception as e:
            logger.error(f"Erreur lors de la recherche (tentative {attempt + 1}/{max_retries + 1}) : {e}")
            
            # Gestion spécifique des erreurs de rate limit
            if "RatelimitException" in str(type(e)):
                if attempt < max_retries:
                    logger.warning("Limite de taux atteinte, nouvelle tentative...")
                    continue
                else:
                    raise HTTPException(
                        status_code=HTTP_429_TOO_MANY_REQUESTS, 
                        detail="Trop de requêtes. Veuillez réessayer plus tard."
                    )
            
            # Pour toute autre erreur
            return "Désolé, je n'ai pas pu trouver de réponse."