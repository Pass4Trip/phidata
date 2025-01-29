import json
import os
import sys 
from typing import List, Optional

# Ajouter le chemin du module logging standard Python avant les imports
sys.path.insert(0, os.path.dirname(os.__file__))
import logging

from dotenv import load_dotenv

# Import correct de PgMemoryDb
from phi.memory.db.postgres import PgMemoryDb
from phi.memory.agent import AgentMemory
from phi.memory.memory import Memory

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Construction dynamique de l'URL de base de données PostgreSQL
def build_postgres_url() -> str:
    """
    Construire dynamiquement l'URL de connexion PostgreSQL à partir des variables d'environnement
    
    Returns:
        str: URL de connexion PostgreSQL
    """
    # Récupérer les paramètres de connexion depuis les variables d'environnement
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

def get_user_preferences(
    query: str,
    user_id: str, 
    db_url: Optional[str] = None, 
    table_name: str = "user_proxy_memories"
) -> str:
    """
    Enrichit une requête en utilisant les mémoires de l'utilisateur.
    
    Args:
        query (str): Requête originale
        user_id (str): Identifiant de l'utilisateur
        db_url (Optional[str]): URL de connexion à la base de données PostgreSQL
        table_name (str): Nom de la table de mémoire
    
    Returns:
        str: Requête enrichie avec les préférences de l'utilisateur
    """
    # Utiliser l'URL de base de données globale si non fournie
    if db_url is None:
        db_url = globals().get('db_url')
    
    if not db_url:
        logger.error("Aucune URL de base de données fournie.")
        return query
    
    try:
        # Créer une mémoire d'agent avec la base de données PostgreSQL
        agent_memory = AgentMemory(
            db=PgMemoryDb(
                table_name=table_name, 
                db_url=db_url
            ),
            # Définir l'ID utilisateur
            user_id=user_id
        )
        
        # Charger les mémoires de l'utilisateur
        memories = agent_memory.db.read_memories(user_id=user_id)
        
        print(f"\n🔍 Nombre de mémoires trouvées : {len(memories)}")
        
        # Stocker les mémoires utilisateur
        user_memories = []
        
        # Parcourir toutes les mémoires
        for memory in memories:
            try:
                memory_content = None
                
                if isinstance(memory.memory, dict):
                    memory_content = memory.memory.get('memory')
                elif isinstance(memory.memory, str):
                    try:
                        memory_dict = json.loads(memory.memory)
                        memory_content = memory_dict.get('memory')
                    except json.JSONDecodeError:
                        memory_content = memory.memory
                else:
                    memory_content = str(memory.memory)
                
                if memory_content:
                    user_memories.append(memory_content)
            
            except Exception as e:
                print(f"❌ Erreur lors du traitement de la mémoire : {e}")
                pass
        
        print(f"\n📋 Nombre de mémoires extraites : {len(user_memories)}")
        
        # Si aucune mémoire, retourner la requête originale
        if not user_memories:
            return query
        
        # Utiliser le modèle LLM pour enrichir la requête
        from phi.agent import Agent
        from phi.llm.openai import OpenAIChat
        
        enrichment_agent = Agent(
            llm=OpenAIChat(model="gpt-4o-mini"),
            instructions=[
                "Tu es un assistant spécialisé dans l'enrichissement de requêtes.",
                "Règles strictes :",
                "1. Comprendre le contexte et les préférences pertinentes de l'utilisateur",
                "2. Enrichir la requête UNIQUEMENT avec des informations générales et personnelles",
                "3. INTERDICTION ABSOLUE d'ajouter des lieux spécifiques non mentionnés dans la requête originale",
                "4. Garder l'intention originale de la requête intacte",
                "5. Ignorer les informations non pertinentes ou obsolètes",
                "6. Répondre UNIQUEMENT par la requête enrichie, sans aucun autre texte",
                "7. Assurer que l'enrichissement apporte une réelle valeur ajoutée",
                "8. Ne jamais inventer ou ajouter des informations fictives",
                "9. Si aucun enrichissement pertinent n'est possible, retourner la requête originale"
            ]
        )
        
        # Préparer le contexte des mémoires
        memories_context = "\n".join(user_memories)
        
        # Enrichir la requête
        enriched_query = enrichment_agent.run(
            f"""
            Mémoires précédentes de l'utilisateur :
            {memories_context}
            
            Requête originale :
            {query}
            
            Instructions :
            - Enrichis la requête SANS ajouter de lieux spécifiques
            - Utilise uniquement les informations générales des mémoires
            - Ne modifie pas la structure de base de la requête
            """
        )
        
        # Extraction directe du contenu
        if isinstance(enriched_query, str):
            print("\n🚀 Requête enrichie :")
            print(enriched_query)
            return enriched_query
        
        # Extraction si c'est un objet avec un attribut 'content'
        if hasattr(enriched_query, 'content'):
            print("\n🚀 Requête enrichie :")
            print(enriched_query.content)
            return enriched_query.content
        
        # Dernier recours
        print("\n🚀 Requête non enrichie, utilisation de la requête originale")
        return query
    
    except Exception as e:
        logger.error(f"Erreur lors de l'enrichissement de la requête : {e}")
        return query

# Exemple d'utilisation
def example_usage():
    # Récupérer l'URL de la base de données depuis une variable d'environnement
    db_url = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/dbname')
    
    # Créer un gestionnaire de préférences
    try:
        # Récupérer les préférences
        query = "Je cherche un restaurant pour ce week end ?"
        enriched_query = get_user_preferences(query, user_id="vinh")
        
        # Si aucune préférence n'est trouvée, afficher un message
        if not enriched_query:
            print("Aucune préférence trouvée.")
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des préférences : {e}")

if __name__ == "__main__":
    example_usage()
