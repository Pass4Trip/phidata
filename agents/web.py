from typing import Optional
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from agents.settings import agent_settings
import os
import logging
from dotenv import load_dotenv
from db.url import get_db_url
from phi.storage.agent.postgres import PgAgentStorage
from phi.agent import AgentMemory
from phi.memory.db.postgres import PgMemoryDb
from utils.logging import configure_logging

# Charger les variables d'environnement
load_dotenv()

# Configuration du logger
configure_logging()
logger = logging.getLogger(__name__)

db_url = get_db_url()

web_searcher_storage = PgAgentStorage(table_name="web_searcher_sessions", db_url=db_url)


def get_web_searcher(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    # Vérifier si la clé API est définie
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("La clé API OpenAI n'est pas définie. Veuillez définir OPENAI_API_KEY dans votre fichier .env.")

    # Configurer le logging en mode DEBUG
    configure_logging('DEBUG')

    # Récupérer l'URL de base de données (optionnel)
    if db_url:
        logger.info(f"URL de base de données configurée : {db_url}")

    return Agent(
        name="Web Searcher",
        agent_id="web-searcher",
        session_id=session_id,
        user_id=user_id,
        # The model to use for the agent
        model=OpenAIChat(
            id=model_id or agent_settings.gpt_4,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        role="Tu es un agent d'une Team qui dispose des capacités à rechercher des informations sur le Web. Tu dois renvoyer tes résultats à Agent Leader",  
        instructions=[
            "Tu es un agent intelligent avec des capacités de recherche web.",
            "Utilise le moteur de recherche DuckDuckGo UNIQUEMENT si :",
            " - La réponse nécessite des informations actuelles ou récentes",
            " - La question demande explicitement une recherche web",
            " - Tes connaissances actuelles sont insuffisantes pour répondre précisément",
            "",
            "Étapes de résolution :",
            "1. Évalue d'abord si une recherche web est vraiment nécessaire",
            "2. Si oui, décompose la question en requêtes de recherche pertinentes",
            "3. Analyse les résultats avec esprit critique",
            "",
            "Règles importantes :",
            " - Privilégie toujours la qualité et la précision des sources",
            " - Si aucune information pertinente n'est trouvée, explique-le clairement",
            " - Garde tes réponses concises et instructives",
            " - En cas d'échec de recherche, utilise tes connaissances existantes",
            " - TOUJOURS mentionner explicitement les informations de contexte utilisateur utilisées",
            "   * Citer les détails spécifiques de l'environnement ou des préférences",
            "   * Expliquer comment ces informations ont influencé ta réponse",
        ],
        tools=[DuckDuckGo()],
        add_datetime_to_instructions=True,
        markdown=True,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable monitoring on phidata.app
        monitoring=True,
        # Show debug logs
        debug_mode=True,  # Passer à True pour activer les logs de débogage
        # Store agent sessions in the database
        storage=web_searcher_storage,

        memory=AgentMemory(
        db=PgMemoryDb(table_name="web_searcher__memory", db_url=db_url), create_user_memories=True, create_session_summary=True),

    )