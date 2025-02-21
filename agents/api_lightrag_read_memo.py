from typing import Optional, Any, Dict, Callable
import os
import logging
from dotenv import load_dotenv
import json
import uuid
from datetime import datetime, timedelta
import requests
from requests import Response
from typing import Dict

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
logger.setLevel(logging.DEBUG)  # Réduire le niveau de log
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

agent_storage_file: str = "orchestrator_agent_sessions.db"



def query_lightrag_execute(question: str, user_id: str = "vinh") -> Dict[str, Any]:
    """
    Effectue une requête au endpoint LightRAG pour obtenir une réponse.
    
    Args:
        question (str): La question à poser
        user_id (str, optional): L'identifiant de l'utilisateur. Defaults to "vinh".
    
    Returns:
        Dict[str, Any]: La réponse du endpoint LightRAG
    """
    url = "http://51.77.200.196:30080/query/"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "question": question,
        "user_id": user_id,
        "vdb_filter": [user_id]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Erreur lors de la requête LightRAG : {e}")
        return {
            "status": "error",
            "content": f"Impossible de contacter le service LightRAG : {str(e)}"
        }



def get_agent_lightrag_read_memo(
    model_id: str = "gpt-4o-mini",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
    stream: bool = False,
    conversation_history: Optional[list] = None,  # Paramètre pour l'historique de session
    **kwargs
) -> Agent:

    # Générer un session_id unique si non fourni
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info(f"🆔 Génération d'un nouvel identifiant de session : {session_id}")    

    # Préparer les instructions initiales
    base_instructions = [
        "Ton nom est Agent LightRAG.",
        "Tu es un agent conversationnel spécialisé dans l'utilisation du service LightRAG.",
        "Tes objectifs sont :",
        "1. Pour CHAQUE requête utilisateur :",
        "   - Utiliser OBLIGATOIREMENT l'outil query_lightrag",
        "   - Passer la requête EXACTEMENT telle qu'elle est reçue",
        "2. Retourner la réponse du endpoint LightRAG SANS MODIFICATION",
        "3. Si le endpoint retourne une erreur :",
        "   - Retourner l'erreur telle quelle",
        "   - Ne PAS essayer de générer une réponse alternative",
        "4. Structure de la réponse :",
        "   - Utiliser EXACTEMENT la structure retournée par LightRAG",
        "   - NE PAS modifier le format ou le contenu",
        "5. Gestion des cas particuliers :",
        "   - En cas d'impossibilité d'utiliser LightRAG, retourner un message d'erreur",
        "   - Ne PAS chercher à contourner ou remplacer le service",
        "6. Principes de base :",
        "   - Transparence totale sur l'utilisation de LightRAG",
        "   - Aucune modification ou interprétation de la réponse",
        "7. Contexte de l'outil :",
        "   - query_lightrag est l'UNIQUE moyen de répondre aux requêtes"
    ]

    def query_lightrag(query: str):
        query_result = query_lightrag_execute(query, user_id)
        # Convertir le résultat en une chaîne de texte lisible
        if isinstance(query_result, dict):
            # Si c'est un dictionnaire, convertir en chaîne JSON lisible
            return json.dumps(query_result, ensure_ascii=False)
        elif isinstance(query_result, str):
            return query_result
        else:
            return str(query_result)

    # Gestion de l'historique de session
    # --------------------------------
    # Exemple d'utilisation de l'historique de conversation
    # L'historique est passé depuis le gestionnaire de session WebSocket
    # Structure attendue : [{'role': 'user'/'assistant', 'content': 'message'}]
    if conversation_history:
        # Ajouter le contexte de la conversation précédente aux instructions
        context_instruction = "Contexte de la conversation précédente :"
        for msg in conversation_history:
            # Traduire le rôle pour plus de clarté
            role = "Utilisateur" if msg['role'] == 'user' else "Assistant"
            context_instruction += f"\n- {role}: {msg['content']}"
        
        # Insérer le contexte après la description initiale
        base_instructions.insert(3, context_instruction)

        # Commentaires sur les possibilités d'utilisation de l'historique :
        # 1. Comprendre le contexte précédent
        # 2. Éviter les répétitions
        # 3. Maintenir la cohérence de la conversation
        # 4. Personnaliser les réponses en fonction des interactions précédentes

    # Créer l'agent Phidata
    agent_lightrag = Agent(
        instructions=base_instructions,
        model=OpenAIChat(
            model=model_id,
            temperature=0.7,
        ),
        #output_format="json",  # Spécifier le format de sortie JSON
        debug_mode=debug_mode,  # Forcer le mode débogage
        agent_id="agent_lightrag",
        user_id=user_id,
        session_id=session_id,
        name="Agent LightRAG",
        tools=[query_lightrag],  # Ajout des outils, dont LightRAG
        memory=AgentMemory(
            db=PgMemoryDb(table_name="web_searcher__memory", db_url=db_url),
            # Commentaires sur les options de mémoire :
            # create_user_memories : Crée des mémoires personnalisées par utilisateur
            # update_user_memories_after_run : Met à jour ces mémoires après chaque exécution
            # create_session_summary : Crée un résumé de la session
            # update_session_summary_after_run : Met à jour ce résumé après chaque exécution
            create_user_memories=True,
            update_user_memories_after_run=True,
            create_session_summary=True,
            update_session_summary_after_run=True,
        ),        
        storage=PgAgentStorage(table_name="web_searcher_sessions", db_url=db_url),
    )

    logger.info("✅ Agent LightRAG initialisé avec succès")
    return agent_lightrag



# Bloc main pour lancer l'agent directement
if __name__ == "__main__":
    import sys
    import argparse
    
    # Configuration de l'analyseur d'arguments
    parser = argparse.ArgumentParser(description="Lancer un agent de base en mode interactif")
    parser.add_argument("--model", default="gpt-4o-mini", help="Modèle OpenAI à utiliser")
    parser.add_argument("--user_id", help="Identifiant utilisateur")
    parser.add_argument("--session_id", help="Identifiant de session")
    

    user_id = "vinh"

    # Analyser les arguments
    args = parser.parse_args()
    
    # Créer l'agent
    agent = get_agent_lightrag_read_memo(
        model_id="gpt-4o-mini", 
        user_id=args.user_id, 
        session_id=args.session_id
    )
    
    # Mode interactif
    print("🤖 Agent LightRAG - Mode Interactif")
    print("Tapez 'exit' ou 'quit' pour quitter.")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("👤 Votre demande : ")
            
            # Obtenir la réponse de l'agent
            print(f"\n🔍 Requête : {user_input}")
            response = agent.run(user_input)

            # Ajouter l'historique de conversation
            conversation_history.append({'role': 'user', 'content': user_input})
            
            # Gérer différents types de réponses
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            
            print(f"\n🤖 Type de réponse : {type(content)}")
            print(f"\n🤖 Contenu de la réponse : {content}")
            
            conversation_history.append({'role': 'assistant', 'content': content})

            # Afficher la réponse
            print("\n🤖 Réponse :", content)
        
        except KeyboardInterrupt:
            print("Au revoir ! 👋")
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            break
