from typing import Optional, Any, Dict, Callable, List, Union
import os
import logging

from bokeh.models import tools
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

def create_dynamic_widget(
    name: Optional[str] = None,
    type: str = 'select', 
    options: Optional[List[str]] = None, 
):
    """
    Génère un widget dynamique pour l'interaction avec l'utilisateur
    
    Args:
        name (str, optional): Nom du widget
        type (str): Type de widget ('select', 'text', 'number')
        options (List[str], optional): Options pour un widget de sélection
    
    Returns:
        Dict représentant la configuration du widget
    """
    widget_config = {
        'name': name or '',
        'type': type,
        'options': options or []
    }
    return widget_config
    
def get_agent_base(
    model_id: str = "gpt-4o-mini",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
    stream: bool = False,
    conversation_history: Optional[list] = None,  # Paramètre pour l'historique de session
    widget_config: Optional[Dict] = None,  # Nouveau paramètre optionnel
    **kwargs
) -> Union[Agent, Dict[str, Any]]:

    # Générer un session_id unique si non fourni
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info(f"🆔 Génération d'un nouvel identifiant de session : {session_id}")    

    def call_create_dynamic_widget():
        widget_config = {
            'name': 'select',
            'type': 'select',
            'options': ['Option 1', 'Option 2'],
        }
        return create_dynamic_widget(widget_config)

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
        "   - 'content': contenu de la réponse ou questions de clarification (TOUJOURS une chaîne)",
        "   - 'metadata': informations supplémentaires (optionnel)",
        "6. Si la question nécessite des précisions :",
        "   - Retourne 'status': 'clarification_needed'",
        "   - Convertis les listes de questions en une chaîne multiligne",
        "7. Si la question nécessite des connaissances spécifiques, utilise les outils à ta disposition",
        "8. Reste toujours professionnel, bienveillant et utile",
        "9. Si tu ne peux pas répondre à une question, explique pourquoi dans le champ 'content'",
        "10. Adapte ton niveau de langage et de détail au contexte de la question",
        "11. Pour chaque interaction, génère une réponse cohérente :",
        "    - Si la requête est claire, fournis une réponse directe et informative",
        "    - Si la requête manque de précision, pose des questions de clarification",
        "    - Assure-toi que la réponse soit toujours lisible et compréhensible",
        "12. Gestion des interactions par widget :",
        "    - Pour un bouton : explique brièvement son contexte ou son utilité",
        "    - Pour une sélection : fournis une réponse adaptée à l'option choisie",
        "13. Transforme TOUJOURS les listes en une chaîne de texte lisible"
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

    # Créer l'agent Phidata
    agent_base = Agent(
        instructions=base_instructions,
        model=OpenAIChat(
            model=model_id,
            temperature=0.7,
            response_format={"type": "json_object"}  # Forcer la réponse JSON
        ),
        #tools=[call_create_dynamic_widget],
        output_format="json",  # Spécifier le format de sortie JSON
        debug_mode=True,  # Forcer le mode débogage
        agent_id="agent_base",
        user_id=user_id,
        session_id=session_id,
        name="Agent Base",
        # memory=AgentMemory(
        #     db=PgMemoryDb(table_name="web_searcher__memory", db_url=db_url),
        #     # Commentaires sur les options de mémoire :
        #     # create_user_memories : Crée des mémoires personnalisées par utilisateur
        #     # update_user_memories_after_run : Met à jour ces mémoires après chaque exécution
        #     # create_session_summary : Crée un résumé de la session
        #     # update_session_summary_after_run : Met à jour ce résumé après chaque exécution
        #     create_user_memories=True,
        #     update_user_memories_after_run=True,
        #     create_session_summary=True,
        #     update_session_summary_after_run=True,
        # ),        
        #storage=PgAgentStorage(table_name="web_searcher_sessions", db_url=db_url),
    )

    logger.debug("✅ Agent de recherche web initialisé avec succès")

    # Création d'une liste de configurations de widgets
    widget_list = []
    
    # Widget de sélection
    widget_select = {
        'name': 'select',
        'type': 'select',
        'options': ['Donne moi la définition de chien', 'Donne moi la définition de chat', 'Explique moi en quelques mot LLM'],
    }
    widget_list.append(widget_select)
    
    # Widget bouton
    widget_button = {
        'name': 'button',
        'type': 'button',
        'button_type': 'primary',
    }
    widget_list.append(widget_button)


    logger.info(f"🔍 Widgets générés : {len(widget_list)} widgets")
    for widget in widget_list:
        logger.info(f"🚀 Configuration du widget : {widget['name']} (type: {widget['type']})")

    return {
        'agent': agent_base,
        'widget_list': widget_list
    }
    
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
    
    # Créer l'agent et récupérer les widgets
    agent_result = get_agent_base(
        model_id="gpt-4o-mini", 
        user_id=args.user_id, 
        session_id=args.session_id
    )
    
    # Extraire l'agent et les widgets
    agent = agent_result['agent']
    widget_list = agent_result['widget_list']
    
    # Afficher les widgets
    print("\n🧩 Widgets disponibles :")
    for widget in widget_list:
        print(f"- {widget['name']} (Type: {widget['type']})")
        if 'options' in widget:
            print(f"  Options: {widget['options']}")
    
    # Mode interactif
    print("\n🤖 Agent Base - Mode Interactif")
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

            print("\n🤖 Réponse :", response)
            print("\n🤖 Widget List :", widget_list)
        
        except KeyboardInterrupt:
            print("\n\nInterruption. Au revoir ! 👋")
            break
        except Exception as e:
            print(f"Erreur : {e}")
            break