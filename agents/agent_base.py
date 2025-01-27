from typing import Optional, Any, Dict, Callable
import os
import logging
from dotenv import load_dotenv
from phi.agent import Agent, AgentMemory
from phi.storage.agent.postgres import PgAgentStorage
import json
from datetime import datetime, timedelta
from phi.model.openai import OpenAIChat
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.memory.db.sqlite import SqliteMemoryDb


# Charger les variables d'environnement
load_dotenv()

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
    model_id: str = "gpt-4o-mini",
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
        debug_mode=True,  # Forcer le mode débogage
        user_id=user_id,
        session_id=session_id,
        name="Agent Base",
        memory=AgentMemory(
            db=SqliteMemoryDb(
                table_name="agent_memory",
                db_file=agent_storage_file,
            ),
            # Create and store personalized memories for this user
            create_user_memories=True,
            # Update memories for the user after each run
            update_user_memories_after_run=True,
            # Create and store session summaries
            create_session_summary=True,
            # Update session summaries after each run
            update_session_summary_after_run=True,
        ),        
        storage=SqlAgentStorage(table_name="agent_sessions", db_file=agent_storage_file),
    )

    logger.debug("✅ Agent de recherche web initialisé avec succès")
    return agent_base

# Exemple d'exécution directe de l'agent
if __name__ == "__main__":
    import sys
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Créer l'agent
    agent = get_agent_base(
        debug_mode=True,
        user_id="paddy",
        session_id="tpaddy_session",
        model_id="gpt-4o-mini",
        stream=False
    )
    
    # Exemple de tâche à exécuter
    def run_agent_test():
        try:
            # Exemples de requêtes de test
            test_queries = [
                "Quelle est ma derniere question ?"
            ]
            
            for query in test_queries:
                print(f"\n{'='*50}")
                print(f"🔍 Requête : {query}")
                print(f"{'='*50}")
                
                # Exécuter l'agent avec la requête
                result = agent.run(query)
                
                print("\n📋 Réponse :")
                print(result)
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution de l'agent : {e}")
            import traceback
            traceback.print_exc()
    
    # Lancer le test
    run_agent_test()