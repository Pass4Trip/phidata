import logging
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from agents.user_proxy import get_user_proxy_agent

# Configurer le logging
logger = logging.getLogger(__name__)

class UserProxyResponse(BaseModel):
    content: str
    status: str = "success"
    metadata: dict = {}

user_proxy_router = APIRouter()

@user_proxy_router.post("/ask", response_model=UserProxyResponse)
async def process_user_proxy_request(
    query: str,
    model_id: str = "gpt-4o-mini",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) :
    """
    Endpoint pour traiter les requêtes via le User Proxy Agent.
    """
    
    try:
        # Initialiser l'agent User Proxy
        user_proxy_agent = get_user_proxy_agent(
            model_id=model_id,
            user_id=user_id, 
            session_id=session_id,
            debug_mode=False
        )
        
        # Exécuter l'agent avec la requête et récupérer la réponse
        try:
            resp_generator = user_proxy_agent.run(query)
            
            # Collecter tous les résultats du générateur
            responses = []
            try:
                for resp in resp_generator:
                    responses.append(str(resp))
            except StopIteration:
                pass
            
            # Construire la réponse
            result_content = " ".join(responses).strip()
            if not result_content:
                result_content = "Aucune réponse générée"
            
            logger.info(f"Résultat obtenu : {result_content}")
        
        except Exception as agent_error:
            # Gérer les erreurs de l'agent
            return UserProxyResponse(
                content=f"Erreur lors de l'exécution de l'agent : {str(agent_error)}",
                status="error",
                metadata={"error_type": type(agent_error).__name__}
            )
        
        return UserProxyResponse(
            content=result_content,
            metadata={
                "model_id": model_id,
                "user_id": user_id,
                "session_id": session_id
            }
        )
    
    except Exception as e:
        logger.exception(f"Erreur globale lors du traitement de la requête : {e}")
