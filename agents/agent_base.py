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
        "Ton nom est AgentBase.",
        "Tu es un agent conversationnel intelligent et polyvalent.",
        "Tu es capable de répondre à une large variété de questions.",
        "Tes objectifs sont :",
        "1. Analyser précisément la requête de l'utilisateur",
        "2. Si la requête manque de précision, demander des éclaircissements",
        "3. En cas de manque de clarté, pose des questions spécifiques pour :",
        "   - Comprendre le contexte exact",
        "   - Préciser les attentes de l'utilisateur",
        "   - Obtenir les informations manquantes",
        "4. Fournir une réponse TOUJOURS au format JSON structuré",
        "5. Structure de la réponse JSON :",
        "   - 'status': 'success', 'clarification_needed' ou 'error'",
        "   - 'content': contenu de la réponse ou questions de clarification",
        "   - 'metadata': informations supplémentaires (optionnel)",
        "6. Si la question nécessite des précisions :",
        "   - Retourne 'status': 'clarification_needed'",
        "   - Liste les questions précises à poser dans 'content'",
        "7. Si la question nécessite des connaissances spécifiques, utilise les outils à ta disposition",
        "8. Reste toujours professionnel, bienveillant et utile",
        "9. Si tu ne peux pas répondre à une question, explique pourquoi dans le champ 'content'",
        "10. Adapte ton niveau de langage et de détail au contexte de la question",
    ]

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
    agent_base = Agent(
        instructions=base_instructions,
        model=OpenAIChat(
            model=model_id,
            temperature=0.7,
            response_format={"type": "json_object"}  # Forcer la réponse JSON
        ),
        output_format="json",  # Spécifier le format de sortie JSON
        debug_mode=debug_mode,  # Forcer le mode débogage
        agent_id="agent_base",
        user_id=user_id,
        session_id=session_id,
        name="Agent Base",
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

    logger.debug("✅ Agent de recherche web initialisé avec succès")
    return agent_base


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
    agent = get_agent_base(
        model_id="gpt-4o-mini", 
        user_id=args.user_id, 
        session_id=args.session_id
    )
    
    # Mode interactif
    print("🤖 Agent Base - Mode Interactif")
    print("Tapez 'exit' ou 'quit' pour quitter.")
    
    conversation_history = []
    
    while True:
        try:
            # Demander une entrée utilisateur
            user_input = input("\n> ")
            
            # Vérifier la sortie
            if user_input.lower() in ['exit', 'quit']:
                print("Au revoir ! 👋")
                break
            
            # Obtenir la réponse de l'agent
            response = agent.run(user_input)

            # Ajouter l'historique de conversation
            conversation_history.append({'role': 'user', 'content': user_input})
            conversation_history.append({'role': 'assistant', 'content': response.content if hasattr(response, 'content') else str(response)})

            # Afficher la réponse
            content = response.content if hasattr(response, 'content') else str(response)
            print("\n🤖 Réponse :", content)
        
        except KeyboardInterrupt:
            print("\n\nInterruption. Au revoir ! 👋")
            break
        except Exception as e:
            print(f"Erreur : {e}")
            break
