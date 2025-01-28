from typing import Optional, Any, Dict, Callable
import os
import logging
from dotenv import load_dotenv
import uuid
import json
from datetime import datetime, timedelta

from phi.model.openai import OpenAIChat
from phi.agent import Agent, AgentMemory
from phi.storage.agent.postgres import PgAgentStorage
from phi.memory.db.postgres import PgMemoryDb


# Importer les nouveaux outils de recherche
from llm_axe.models import llm_axe_OpenAIChat
from llm_axe.agents import OnlineAgent


# Charger les variables d'environnement
load_dotenv()

# Construction dynamique de l'URL de base de données PostgreSQL
def build_postgres_url():
    """
    Construire dynamiquement l'URL de connexion PostgreSQL à partir des variables d'environnement
    
    Returns:
        str: URL de connexion PostgreSQL
    """
    db_host = os.getenv('DB_HOST', 'vps-af24e24d.vps.ovh.net')
    db_port = os.getenv('DB_PORT', '30030')
    db_name = os.getenv('DB_NAME', 'myboun')
    db_user = os.getenv('DB_USER', 'p4t')
    db_password = os.getenv('DB_PASSWORD', '')
    db_schema = os.getenv('DB_SCHEMA', 'ai')
    
    # Construire l'URL de connexion PostgreSQL avec le schéma
    db_url = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?options=-c%20search_path%3D{db_schema}'
    
    return db_url

# Générer l'URL de base de données
db_url = build_postgres_url()

# Configuration du logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Réduire le niveau de log
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

agent_storage_file: str = "orchestrator_agent_sessions.db"



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

    # Générer un session_id unique si non fourni
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info(f"🆔 Génération d'un nouvel identifiant de session : {session_id}")    

    # Créer un outil de recherche web personnalisé
    def web_search_tool(query: str):
        logger.debug(f" Préparation de la recherche web pour la requête : {query}")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(" Clé API OpenAI manquante")
            raise ValueError("Clé API OpenAI manquante. Veuillez la définir dans le fichier .env")

        logger.debug(" Initialisation du modèle LLM")
        llm = llm_axe_OpenAIChat(api_key=api_key)
        logger.debug(f" Modèle LLM initialisé : {llm}")
        
        logger.debug(" Création de l'agent de recherche en ligne")
        searcher = OnlineAgent(llm, stream=False)
        logger.debug(f" Agent de recherche créé : {searcher}")
        
        def run(task):
            logger.debug(f" Démarrage de la recherche web pour la tâche : {task}")
            try:
                result = searcher.search(task)
                logger.debug(f" Recherche web terminée avec succès")
                logger.debug(f" Résultats de la recherche : {len(result)} éléments")
                return result
            except Exception as e:
                logger.error(f" Erreur lors de la recherche web : {e}")
                raise
        
        # Remplacer la méthode run de l'agent
        searcher.run = run
        
        logger.info(" Exécution de la recherche web")
        res = searcher.search(query)
        
        logger.debug(f" Résultats de la recherche : {len(res)} éléments")
        logger.info(f" Détails des résultats : {res}")
        
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
        debug_mode=True,  # Forcer le mode débogage
        user_id=user_id,
        session_id=session_id,
        name="Web Search Agent",
        memory=AgentMemory(
            db=PgMemoryDb(table_name="agent_memories", db_url=db_url),
            # Create and store personalized memories for this user
            create_user_memories=True,
            # Update memories for the user after each run
            update_user_memories_after_run=True,
            # Create and store session summaries
            create_session_summary=True,
            # Update session summaries after each run
            update_session_summary_after_run=True,
        ),        
        storage=PgAgentStorage(table_name="agent_sessions", db_url=db_url),
    )

    logger.debug(" Agent de recherche web initialisé avec succès")
    return web_agent