from typing import Optional
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.python import PythonTools
from agents.settings import agent_settings
import os
import logging
from dotenv import load_dotenv
from db.url import get_db_url
from phi.storage.agent.postgres import PgAgentStorage
from phi.agent import AgentMemory
from phi.memory.db.postgres import PgMemoryDb
from utils.colored_logging import get_colored_logger
import httpx
import logging
from typing import Optional, Dict, Any, List

# Charger les variables d'environnement
load_dotenv()

# Configuration du logger
logger = get_colored_logger('agents.api_knowledge', 'APIKnowledgeAgent', level=logging.INFO)

db_url = get_db_url()

api_knowledge_storage = PgAgentStorage(table_name="api_knowledge_sessions", db_url=db_url)

# Configuration de l'URL de l'API LightRAG
API_LIGHTRAG_QUERY_URL = 'http://51.77.200.196:30080/query/'

def sync_query_lightrag_api(
    question: str, 
    user_id: str = "vinh", 
    mode: str = "hybrid", 
    vdb_filter: Optional[List[Dict]] = None, 
    limit: int = 10, 
    offset: int = 0
) -> str:
    """
    Interroger l'API de requête de LightRAG de manière synchrone
    
    Args:
        question (str): Question à soumettre
        user_id (str, optional): ID de l'utilisateur. Défaut à "vinh".
        mode (str, optional): Mode de requête. Défaut à "hybrid".
        vdb_filter (Optional[List[Dict]], optional): Filtres VDB. Défaut à None.
        limit (int, optional): Nombre maximum de résultats. Défaut à 10.
        offset (int, optional): Décalage des résultats. Défaut à 0.
    
    Returns:
        str: Texte de réponse de la requête
    """

    # Log coloré pour indiquer la prise en charge de la demande
    logger.info("📚 Agent Connaissance API prêt à interroger les bases de connaissances")
    
    # Utiliser une liste vide par défaut si vdb_filter est None
    if vdb_filter is None:
        vdb_filter = []

    try:
        # Configuration du client avec timeouts généreux
        with httpx.Client(
            timeout=httpx.Timeout(
                connect=10.0,   # Timeout de connexion
                read=60.0,      # Timeout de lecture long
                write=60.0,     # Timeout d'écriture long
                pool=60.0       # Timeout du pool de connexions
            ),
            follow_redirects=True,  # Activer le suivi des redirections
            max_redirects=3  # Limiter le nombre de redirections
        ) as client:
            # Préparer la payload de requête
            payload = {
                "question": question,
                "user_id": user_id,
                "mode": mode,
                "vdb_filter": vdb_filter,
                "limit": limit,
                "offset": offset
            }
            
            # Log détaillé de la requête
            logger.info(f" Requête LightRAG - Question : {question}")
            logger.debug(f" Paramètres de requête : {payload}")
            
            try:
                response = client.post(
                    API_LIGHTRAG_QUERY_URL, 
                    json=payload
                )
                
                # Log détaillé de la réponse
                logger.debug(f" Requête HTTP: {response.request.method} {response.request.url}")
                logger.debug(f" Réponse HTTP: {response.status_code} {response.reason_phrase}")
                
                # Récupérer et log du contenu de la réponse
                response_content = response.json()
                logger.debug(f" Contenu de la réponse : {response_content}")
                
                if response.status_code == 200 and response_content.get('status') == 'success':
                    # Extraire et retourner la réponse textuelle
                    response_text = response_content.get('response', 'Aucune réponse disponible')
                    
                    logger.debug(f" Réponse extraite : {response_text}")
                    return response_text
                else:
                    logger.error(f" Échec de la requête : {response.status_code} - {response.text}")
                    return "Erreur lors de la requête à l'API"
            
            except httpx.TimeoutException as e:
                logger.error(f" Timeout lors de la requête : {e}")
                return "Timeout lors de la requête"
            except ValueError as e:
                logger.error(f" Erreur de parsing JSON : {e}")
                return "Erreur de parsing de la réponse"
    
    except Exception as e:
        logger.error(f" Erreur inattendue lors de la requête à l'API : {e}")
        import traceback
        logger.error(traceback.format_exc())
        return "Erreur inattendue lors de la requête"

def get_api_knowledge_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False
):
    """
    Créer un agent de connaissance API utilisant LightRAG
    """
    from phi.agent import Agent
    from phi.model.openai import OpenAIChat
    from phi.tools.python import PythonTools
    from agents.settings import agent_settings

    # Récupérer l'URL de base de données (optionnel)
    if db_url:
        logger.debug(f"URL de base de données configurée : {db_url}")



    # Créer l'agent de connaissance
    api_knowledge_agent = Agent(
        name="API Knowledge Agent",
        role="Agent spécialisé dans l'interrogation de bases de connaissances via LightRAG",
        model=OpenAIChat(
            id=agent_settings.gpt_4,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        instructions=[
            "Tu es un agent expert dans l'interrogation de bases de connaissances structurées.",
            "Ton rôle est d'interpréter précisément les requêtes et de les traduire en requêtes API efficaces.",
            "Utilise la fonction sync_query_lightrag_api pour rechercher des informations.",
            "Étapes de traitement :",
            "1. Analyse soigneusement la requête reçue",
            "2. Utilise sync_query_lightrag_api pour obtenir les informations",
            "3. Si la requête échoue, fournis un message d'erreur clair",
            "4. Formate la réponse de manière lisible et concise",
            "",
            "Recommandations :",
            "- Adapte le mode de requête selon le contexte",
            "- Limite le nombre de résultats pour optimiser la réponse",
            "- Priorise la précision et la pertinence des informations"
        ],
        tools=[
            PythonTools(),
            sync_query_lightrag_api
        ],
        show_tool_calls=True,
        debug_mode=debug_mode,
        user_id=user_id,
        session_id=session_id,
        markdown=True,
        stream=False  # Désactiver le streaming
    )

    return api_knowledge_agent

# Exemple d'utilisation
if __name__ == "__main__":
    test_query = "Qu'est-ce que l'intelligence artificielle ?"
    api_knowledge_agent = get_api_knowledge_agent()
    result = api_knowledge_agent.print_response(test_query)