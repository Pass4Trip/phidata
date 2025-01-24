from typing import Optional, Any, Dict, Callable
import os
import logging
from dotenv import load_dotenv
from phi.agent import Agent
from phi.llm.openai.chat import OpenAIChat
from phi.storage.agent.postgres import PgAgentStorage
import json
from datetime import datetime, timedelta

# Importer les nouveaux outils de recherche
from llm_axe.models import OpenAIChat
from llm_axe.agents import OnlineAgent

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logger = logging.getLogger(__name__)


def get_web_searcher(
    model_id: str = "gpt-4o-mini",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
    stream: bool = False,
    **kwargs
) -> Agent:
    """
    Crée un agent de recherche web utilisant llm-axe OnlineAgent.
    
    Args:
        model_id (str): L'identifiant du modèle OpenAI à utiliser.
        user_id (Optional[str]): L'identifiant de l'utilisateur.
        session_id (Optional[str]): L'identifiant de session.
        debug_mode (bool): Mode de débogage activé.
    
    Returns:
        Agent: Un agent de recherche web configuré.
    """

    # Créer un outil de recherche web personnalisé
    def web_search_tool(query: str = "web search"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Clé API OpenAI manquante. Veuillez la définir dans le fichier .env")

        llm = OpenAIChat(api_key=api_key)
        searcher = OnlineAgent(llm, stream=False)
        res = searcher.search(query)

        json_res = []
        json_res.append(res)
        
        return json.dumps(json_res) 

    # Créer l'agent Phidata
    web_agent = Agent(
        tools=[web_search_tool],
        instructions=[
            "Tu es un agent de recherche web intelligent.",
            "Pour effectuer des recherches sur le web, tu dois TOUJOURS utiliser l'outil web_search_tool.",
            "Ton objectif est de trouver des informations précises et pertinentes.",
            "Voici comment procéder:",
            "1. Reçois une requête de recherche",
            "2. Si la requête contient des références temporelles relatives (ex: 'demain', 'dans une semaine', 'hier'):",
            "   - Utilise la date actuelle (" + datetime.now().strftime("%Y-%m-%d") + ") comme référence",
            "   - Convertis ces références en dates précises avant d'effectuer la recherche",
            "   - Exemple: 'événements de demain' -> 'événements du " + (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d") + "'",
            "3. Utilise web_search_tool avec cette requête pour accéder au web",
            "4. Analyse les résultats retournés par web_search_tool",
            "5. Fournis une réponse concise et informative basée sur ces résultats",
            "Si tu ne trouves pas d'informations via web_search_tool, explique pourquoi."
        ],
        debug_mode=debug_mode,
        user_id=user_id,
        session_id=session_id,
        name="Web Search Agent",
        stream=False
    )

    return web_agent