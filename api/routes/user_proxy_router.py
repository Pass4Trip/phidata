import logging
from typing import Optional
from datetime import datetime
import random

from fastapi import APIRouter, Request
from pydantic import BaseModel
import openai

from agents.user_proxy import get_user_proxy_agent

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajouter un handler pour la console si nécessaire
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

class UserProxyResponse(BaseModel):
    content: str
    status: str = "success"
    metadata: dict = {}

def get_waiting_message(query: str) -> str:
    """Génère un message d'attente personnalisé et amusant en fonction de la requête."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": """Tu es un assistant très sympathique et drôle.
                Ta mission est de générer des réponses courtes et amusantes pour dire à l'utilisateur que sa demande est en cours de traitement.
                - Utilise un ton léger et amical
                - Ajoute des emojis appropriés
                - Fais référence au contenu de sa requête de manière humoristique
                - Garde la réponse concise (max 150 caractères)
                - Évite d'être trop technique
                - Varie les réponses pour ne pas être répétitif"""},
                {"role": "user", "content": f"""La requête de l'utilisateur est : "{query}"
                Génère une réponse amusante pour lui dire que je réfléchis à sa demande."""}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Erreur lors de la génération du message d'attente: {e}")
        return "🤔 Hmm, laissez-moi réfléchir à ça... Je consulte mes petits neurones et je reviens vers vous avec une super réponse ! ✨"

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
            response = user_proxy_agent.run(query)
            
            # Extraire le contenu du message de l'assistant
            messages = response.messages
            assistant_message = next(msg for msg in messages if msg.role == 'assistant')
            #logger.info(f"Message complet de l'assistant: {assistant_message}")

            # Si l'assistant utilise un tool_call (comme submit_task)
            if assistant_message.tool_calls and not assistant_message.content:
                # Vérifier si c'est un submit_task
                tool_call = assistant_message.tool_calls[0]
                if tool_call['function']['name'] == 'submit_task':
                    result_content = get_waiting_message(query)
            else:
                result_content = assistant_message.content or "Aucune réponse générée"
            
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
