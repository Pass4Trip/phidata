from typing import Optional, Any, Dict, Callable
import os
import logging
from dotenv import load_dotenv
import json
import uuid
from datetime import datetime, timedelta

from phi.agent import Agent, AgentMemory
from phi.model.openai import OpenAIChat
from phi.storage.agent.postgres import PgAgentStorage
from phi.memory.db.postgres import PgMemoryDb


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

model_id = os.getenv('model_id', 'gpt-4o-mini')

# Configuration du logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Réduire le niveau de log
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

agent_storage_file: str = "orchestrator_agent_sessions.db"

def get_agent_base(
    #model_id: str = "gpt-4o-mini",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
    stream: bool = False,
    **kwargs
) -> Agent:
    """
    Crée un agent conversationnel de base polyvalent.
    
    Cet agent est conçu pour :
    - Répondre à une variété de questions
    - Fournir des informations claires et concises
    - S'adapter à différents contextes de conversation
    
    Args:
        model_id (str): L'identifiant du modèle OpenAI à utiliser.
        user_id (Optional[str]): L'identifiant de l'utilisateur.
        session_id (Optional[str]): L'identifiant de session.
        debug_mode (bool): Mode de débogage activé.
    
    Returns:
        Agent: Un agent conversationnel polyvalent configuré.
    """

    # Générer un session_id unique si non fourni
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info(f"🆔 Génération d'un nouvel identifiant de session : {session_id}")    

    # Créer l'agent Phidata
    agent_base = Agent(
        instructions=[
            "Tu es un agent conversationnel intelligent et polyvalent.",
            "Tu es capable de répondre à une large variété de questions.",
            "Tes objectifs sont :",
            "1. Comprendre précisément la requête de l'utilisateur",
            "2. Fournir une réponse claire, concise et informative",
            "3. Si la question nécessite des connaissances spécifiques, utilise les outils à ta disposition",
            "4. Reste toujours professionnel, bienveillant et utile",
            "5. Si tu ne peux pas répondre à une question, explique pourquoi de manière constructive",
            "6. Adapte ton niveau de langage et de détail au contexte de la question"
        ],
        model_id=model_id,
        debug_mode=debug_mode,  # Forcer le mode débogage
        agent_id="agent_base",
        user_id=user_id,
        session_id=session_id,
        name="Agent Base",
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

    logger.debug("✅ Agent de recherche web initialisé avec succès")
    return agent_base
