from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional

from agents.user_proxy import get_user_proxy_agent, UserProxyAgent

# Création du routeur
user_proxy_router = APIRouter()

# Modèle de requête
class UserProxyRequest(BaseModel):
    request: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

# Modèle de réponse
class UserProxyResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

# Singleton pour l'agent
_user_proxy_agent: Optional[UserProxyAgent] = None

def get_user_proxy_singleton():
    """
    Obtient ou crée une instance singleton de l'agent UserProxy.
    """
    global _user_proxy_agent
    if _user_proxy_agent is None:
        _user_proxy_agent = get_user_proxy_agent()
    return _user_proxy_agent

@user_proxy_router.post("/process", response_model=UserProxyResponse)
async def process_user_request(
    request: UserProxyRequest, 
    agent: UserProxyAgent = Depends(get_user_proxy_singleton)
):
    """
    Endpoint pour traiter une requête utilisateur via l'agent UserProxy.
    """
    try:
        # Traitement de la requête
        result = agent.route_request(
            request=request.request,
            user_id=request.user_id,
            context=request.context
        )

        return UserProxyResponse(
            status="success", 
            message="Requête traitée avec succès",
            data=result
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors du traitement de la requête : {str(e)}"
        )

@user_proxy_router.get("/health")
async def health_check():
    """
    Endpoint de vérification de santé de l'agent.
    """
    return {"status": "healthy", "service": "UserProxyAgent"}
