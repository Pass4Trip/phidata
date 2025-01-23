from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, Dict, Any, Union, List
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_429_TOO_MANY_REQUESTS
from pydantic import BaseModel, Field
import logging
import json
import re

from agents.orchestrator import process_user_request

logger = logging.getLogger(__name__)
orchestrator_router = APIRouter()

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

class OrchestratorResponse(BaseModel):
    """
    Modèle structuré pour la réponse de l'orchestrateur
    """
    query: str = Field(..., description="La requête originale de l'utilisateur")
    result: Union[str, List[str], Dict[str, Any]] = Field(..., description="Résultat de la requête")
    agent_used: Optional[str] = Field(None, description="Agent utilisé pour traiter la requête")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées supplémentaires")
    error: Optional[str] = Field(None, description="Message d'erreur si le traitement a échoué")

@orchestrator_router.post("/process")
async def route_and_process_request(
    query: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False
) -> OrchestratorResponse:
    """
    Endpoint pour router et traiter une requête utilisateur via l'agent principal.
    
    Args:
        query (str): La requête de l'utilisateur
        user_id (Optional[str]): ID de l'utilisateur
        session_id (Optional[str]): ID de session
        debug_mode (bool): Mode de débogage
    
    Returns:
        OrchestratorResponse: Réponse structurée du traitement de la requête
    """
    try:
        # Traiter la requête avec l'agent routeur principal
        result = await process_user_request(
            user_request=query,
            user_id=user_id,
            session_id=session_id,
            debug_mode=debug_mode
        )
        
        # Convertir le résultat si nécessaire
        if isinstance(result, dict):
            result = str(result)
        elif isinstance(result, list):
            result = "\n".join(map(str, result))
        
        # Nettoyer et structurer la réponse
        cleaned_result = clean_response(str(result)) if result is not None else "Aucun résultat"
        
        return OrchestratorResponse(
            query=query,
            result=cleaned_result,
            agent_used="Multi-Purpose Intelligence Team",
            metadata={
                "debug_mode": debug_mode,
                "user_id": user_id,
                "session_id": session_id
            }
        )
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête : {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Erreur lors du traitement : {str(e)}"
        )