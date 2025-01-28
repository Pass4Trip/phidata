import logging
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from agents.user_proxy import get_user_proxy_agent

# Configurer le logging
logger = logging.getLogger(__name__)



user_proxy_router = APIRouter()

@user_proxy_router.post("/ask")
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
            result_content = ""
            try:
                for resp in resp_generator:
                    result_content += str(resp) + "\n"
            except StopIteration:
                pass
            
            # Nettoyer le résultat
            result_content = result_content.strip()
            if not result_content:
                result_content = "Aucune réponse générée"
            
            logger.info(f"Résultat obtenu : {result_content}")
        
        except Exception as agent_error:
            # Gérer les erreurs de l'agent
            result_content = f"Erreur lors de l'exécution de l'agent : {str(agent_error)}"
            logger.error(result_content)
        
        return result_content
    
    except Exception as e:
        logger.exception(f"Erreur globale lors du traitement de la requête : {e}")
        
        return result_content
