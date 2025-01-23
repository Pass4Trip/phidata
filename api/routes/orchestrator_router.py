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
    """
    try:
        # Traiter la requête via l'orchestrateur
        result = await process_user_request(
            user_request=query,
            debug_mode=debug_mode
        )
        
        # Gérer les résultats None ou vides
        result_value = result.get('result', '')
        if result_value is None:
            result_value = ''
        
        # Nettoyer la réponse si c'est une chaîne
        if isinstance(result_value, str):
            result_value = clean_response(result_value)
        elif isinstance(result_value, (list, dict)):
            result_value = str(result_value)
        
        return OrchestratorResponse(
            query=query,
            result=result_value,
            agent_used=result.get('agent_used', 'Multi-Purpose Intelligence Team'),
            metadata={
                'user_id': user_id,
                'session_id': session_id,
                **result.get('metadata', {})
            }
        )
        
    except Exception as e:
        logger.exception(f"Erreur lors du traitement de la requête : {e}")
        return OrchestratorResponse(
            query=query,
            result="",
            error=str(e),
            metadata={
                'user_id': user_id,
                'session_id': session_id
            }
        )