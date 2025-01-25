import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.lazy_imports import lazy_import

# Import paresseux de l'agent
get_user_proxy_agent = lazy_import('agents.user_proxy', 'get_user_proxy_agent')

# Configuration du logging
logger = logging.getLogger(__name__)

# Routeur pour l'agent UserProxy
user_proxy_router = APIRouter()

class UserProxyRequest(BaseModel):
    """Modèle de requête pour l'agent UserProxy"""
    query: str = Field(..., description="Requête de l'utilisateur")
    context: Optional[Dict[str, Any]] = Field(None, description="Contexte supplémentaire")
    user_id: Optional[str] = Field(None, description="ID de l'utilisateur")
    session_id: Optional[str] = Field(None, description="ID de session")

class UserProxyResponse(BaseModel):
    """Modèle de réponse de l'agent UserProxy"""
    status: str
    target_agent: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@user_proxy_router.post("/route", response_model=UserProxyResponse)
async def route_user_request(request: UserProxyRequest):
    """
    Route une requête utilisateur vers l'agent approprié.
    
    Args:
        request (UserProxyRequest): Requête de l'utilisateur
    
    Returns:
        UserProxyResponse: Résultat du routage
    """
    try:
        # Initialisation de l'agent UserProxy
        user_proxy_agent = get_user_proxy_agent(
            model="gpt-4o-mini", 
            debug_mode=False
        )
        
        # Routage de la requête
        routing_result = await user_proxy_agent.route_request(
            request.query, 
            context=request.context
        )
        
        return UserProxyResponse(**routing_result)
    
    except Exception as e:
        logger.error(f"Erreur lors du routage de la requête : {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur de traitement : {str(e)}"
        )

@user_proxy_router.post("/clarify", response_model=UserProxyResponse)
async def request_clarification(request: UserProxyRequest):
    """
    Demande de clarification pour une requête.
    
    Args:
        request (UserProxyRequest): Requête nécessitant une clarification
    
    Returns:
        UserProxyResponse: Résultat de la demande de clarification
    """
    try:
        # Initialisation de l'agent UserProxy
        user_proxy_agent = get_user_proxy_agent(
            model="gpt-4o-mini", 
            debug_mode=False
        )
        
        # Demande de clarification
        clarification_response = await user_proxy_agent.handle_clarification_request(
            request.query, 
            clarification_needed="Informations supplémentaires requises"
        )
        
        return UserProxyResponse(
            status="clarification_needed",
            response={"clarification_message": clarification_response}
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de la demande de clarification : {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur de clarification : {str(e)}"
        )
