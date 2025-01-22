from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, Dict, Any
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_429_TOO_MANY_REQUESTS
from pydantic import BaseModel, Field
import logging
import json
import re

from agents.api_knowledge import get_api_knowledge_agent

logger = logging.getLogger(__name__)
api_knowledge_router = APIRouter()

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

class APIKnowledgeResponse(BaseModel):
    """
    Modèle structuré pour la réponse de recherche de connaissances
    """
    query: str = Field(..., description="La requête originale de l'utilisateur")
    result: str = Field(..., description="Résultat de la recherche ou de l'analyse")
    sources: Optional[list[str]] = Field(None, description="Sources utilisées pour générer la réponse")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées supplémentaires")
    error: Optional[str] = Field(None, description="Message d'erreur si la recherche a échoué")

@api_knowledge_router.get("/knowledge")
async def api_knowledge_search(
    query: str,
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    max_retries: int = 2
) -> APIKnowledgeResponse:
    """
    Endpoint pour effectuer une recherche de connaissances via l'agent API Knowledge.
    
    Args:
        query (str): La requête de recherche
        model_id (Optional[str]): ID du modèle à utiliser
        user_id (Optional[str]): ID de l'utilisateur
        session_id (Optional[str]): ID de session
        max_retries (int): Nombre maximum de tentatives en cas d'échec
    
    Returns:
        APIKnowledgeResponse: Réponse structurée de la recherche
    """
    try:
        # Initialiser l'agent de connaissances
        api_knowledge_agent = get_api_knowledge_agent(
            model_id=model_id,
            user_id=user_id,
            session_id=session_id
        )
        
        # Effectuer la recherche
        result = api_knowledge_agent.print_response(query, stream=False)
        
        # Nettoyer et structurer la réponse
        cleaned_result = clean_response(str(result))
        
        return APIKnowledgeResponse(
            query=query,
            result=cleaned_result,
            sources=[],  # À implémenter selon les capacités de l'agent
            metadata={
                "agent_name": api_knowledge_agent.name,
                "model_used": api_knowledge_agent.model.id
            }
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de la recherche de connaissances : {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Erreur lors de la recherche : {str(e)}"
        )
