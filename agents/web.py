from typing import Optional
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.googlesearch import GoogleSearch
from agents.settings import agent_settings
import os
import logging
from dotenv import load_dotenv
from db.url import get_db_url
from phi.storage.agent.postgres import PgAgentStorage
from phi.agent import AgentMemory
from phi.memory.db.postgres import PgMemoryDb
from utils.colored_logging import get_colored_logger

# Charger les variables d'environnement
load_dotenv()

# Configuration du logger
logger = get_colored_logger('agents.web', 'WebAgent', level=logging.INFO)

db_url = get_db_url()

web_searcher_storage = PgAgentStorage(table_name="web_searcher_sessions", db_url=db_url)

class EnhancedWebSearchTool(GoogleSearch):
    """
    Outil de recherche web amélioré basé sur Google Search
    """
    def web_search(
        self, 
        query: str, 
        max_results: int = 5,
        language: str = 'en'
    ) -> str:
        """
        Recherche web avec gestion des erreurs améliorée
        """
        # Log coloré pour indiquer la prise en charge de la demande
        logger.info("🌐 Agent Web prêt à traiter les requêtes de recherche web")
        
        try:
            logger.info(f"Recherche web pour la requête : {query}")
            results = self.google_search(query=query, max_results=max_results, language=language)
            logger.debug(f"Résultats de recherche obtenus : {len(results)} résultats")
            return results
        except Exception as e:
            logger.error(f"Erreur lors de la recherche web : {e}")
            return f"Erreur de recherche : {e}. Impossible de récupérer les résultats."

def get_web_searcher(
    model_id: Optional[str] = None,
    storage: Optional[PgAgentStorage] = web_searcher_storage,
    memory: Optional[AgentMemory] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    # Vérifier si la clé API est définie
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("La clé API OpenAI n'est pas définie. Veuillez définir OPENAI_API_KEY dans votre fichier .env.")

    # Récupérer l'URL de base de données (optionnel)
    if db_url:
        logger.debug(f"URL de base de données configurée : {db_url}")

    web_agent = Agent(
        name="Web Search Agent",
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
            "Perform web searches to find the most relevant and up-to-date information",
            "Always include sources for the information",
            "Provide clear and concise summaries",
            "",
            "Tu es un agent intelligent avec des capacités de recherche web.",
            "Utilise le moteur de recherche Google UNIQUEMENT si :",
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
        tools=[EnhancedWebSearchTool()],
        add_datetime_to_instructions=True,
        markdown=True,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable monitoring on phidata.app
        monitoring=True,
        # Show debug logs
        debug_mode=debug_mode,
        # Store agent sessions in the database
        storage=storage,

        memory=AgentMemory(
            db=PgMemoryDb(table_name="web_searcher__memory", db_url=db_url), 
            create_user_memories=True, 
            create_session_summary=True
        ),
        stream=False,  # Désactiver le streaming
    )

    def web_search(query):
        """
        Méthode utilitaire pour effectuer une recherche web
        """
        return web_agent.print_response(query, stream=False)

    return web_agent