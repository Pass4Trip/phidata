from typing import Optional, List
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.python import PythonTools
from agents.settings import agent_settings
import os
import logging
from dotenv import load_dotenv
from db.url import get_db_url
from phi.storage.agent.postgres import PgAgentStorage
from phi.agent import AgentMemory
from phi.memory.db.postgres import PgMemoryDb
from utils.colored_logging import get_colored_logger
import io
import sys
import re

# Importer les agents spécialisés
from agents.web import get_web_searcher
from agents.api_knowledge import get_api_knowledge_agent
from agents.data_analysis import get_data_analysis_agent

# Charger les variables d'environnement
load_dotenv()

# Configuration du logger
logger = get_colored_logger('agents.main_router_agent', 'MainRouterAgent', level=logging.DEBUG)

db_url = get_db_url()

def clean_ansi_escape(text):
    """
    Nettoie les codes d'échappement ANSI d'une chaîne de caractères
    """
    # Expression régulière pour supprimer les codes ANSI
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def extract_response_content(text):
    """
    Extrait le contenu textuel de la réponse de l'agent
    """
    # Nettoyer les codes ANSI
    clean_text = clean_ansi_escape(text)
    
    # Extraire le contenu entre "Résumé :" et "Source :"
    resume_match = re.search(r'Résumé :(.*?)Source :', clean_text, re.DOTALL)
    if resume_match:
        return resume_match.group(1).strip()
    
    # Si pas de résumé, retourner le texte nettoyé
    return clean_text.strip()

def create_agent_team(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False
) -> Agent:
    """
    Créer une équipe d'agents avec des rôles spécifiques
    """
    # Vérifier si la clé API est définie
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("La clé API OpenAI n'est pas définie. Veuillez définir OPENAI_API_KEY dans votre fichier .env.")

    # Configurer le logging en mode DEBUG
    logger.setLevel(logging.INFO)

    # Récupérer l'URL de base de données (optionnel)
    if db_url:
        logger.debug(f"URL de base de données configurée : {db_url}")

    # Créer les agents spécialisés
    web_agent = get_web_searcher(
        user_id=user_id, 
        session_id=session_id, 
        debug_mode=debug_mode
    )
    web_agent.name = "Web Research Agent"
    web_agent.role = "Rechercher des informations actuelles et pertinentes sur le web"

    api_knowledge_agent = get_api_knowledge_agent(
        user_id=user_id, 
        session_id=session_id, 
        debug_mode=debug_mode
    )
    api_knowledge_agent.name = "Knowledge API Agent"
    api_knowledge_agent.role = "Accéder et analyser des connaissances via des API spécialisées"

    data_analysis_agent = get_data_analysis_agent(
        user_id=user_id, 
        session_id=session_id, 
        debug_mode=debug_mode
    )
    data_analysis_agent.name = "Data Analysis Agent"
    data_analysis_agent.role = "Effectuer des analyses approfondies et générer des insights à partir de données"

    # Agent principal de routage et de coordination
    router_agent = Agent(
        name="Main Router Agent",
        role="Coordonner et router les requêtes entre les différents agents spécialisés",
        model=OpenAIChat(
            id=agent_settings.gpt_4,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        instructions=[
            "Analyser soigneusement chaque requête utilisateur",
            "Déterminer la meilleure stratégie d'utilisation des agents spécialisés",
            "Coordonner les interactions entre les agents",
            "Compiler et synthétiser les résultats des différents agents",
            "Fournir une réponse claire et complète",
        ],
        tools=[PythonTools()],
        show_tool_calls=True,
        markdown=True,
        stream=False  # Désactiver le streaming
    )
    logger.info("🔀 Agent Routeur Principal prêt à coordonner les requêtes")

    # Créer l'équipe d'agents
    agent_team = Agent(
        name="Multi-Purpose Intelligence Team",
        team=[
            web_agent, 
            api_knowledge_agent, 
            data_analysis_agent, 
            router_agent
        ],
        instructions=[
            "Collaborer efficacement pour résoudre la requête de l'utilisateur",
            "Chaque agent doit utiliser ses compétences spécifiques",
            "Partager les informations et les insights entre les agents",
            "Être transparent sur la source et la méthode d'obtention des informations",
        ],
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
        monitoring=True,
        debug_mode=debug_mode,
        storage=PgAgentStorage(table_name="multi_agent_team_sessions", db_url=db_url),
        memory=AgentMemory(
            db=PgMemoryDb(table_name="multi_agent_team_memory", db_url=db_url),
            create_user_memories=True,
            create_session_summary=True
        ),
    )

    return agent_team

def process_user_request(
    user_request: str, 
    user_id: Optional[str] = None, 
    session_id: Optional[str] = None, 
    debug_mode: bool = False
):
    """
    Traiter une requête utilisateur avec l'équipe d'agents
    """
    try:
        agent_team = create_agent_team(
            user_id=user_id, 
            session_id=session_id, 
            debug_mode=debug_mode
        )
        
        # Exécuter la requête avec l'équipe d'agents
        response = agent_team.run(user_request, stream=False)
        
        # Log détaillé du type de réponse
        #logger.debug(f"Type de réponse reçue : {type(response)}")
        #logger.debug(f"Contenu de la réponse : {response}")
        
        # Vérifier et convertir la réponse
        if response is None:
            logger.warning("Aucune réponse générée par l'agent.")
            return "Aucun résultat n'a pu être généré."
        
        # Convertir la réponse en chaîne lisible
        if isinstance(response, dict):
            # Log détaillé du dictionnaire
            logger.debug(f"Clés du dictionnaire : {list(response.keys())}")
            
            # Gestion spécifique des réponses de l'API Knowledge
            if response.get('status') == 'success':
                # Extraction des résultats
                if 'results' in response:
                    results = response.get('results', [])
                    # Log du nombre de résultats
                    logger.debug(f"Nombre de résultats : {len(results)}")
                    
                    # Formater les résultats en une liste de textes
                    formatted_results = "\n".join([
                        str(result.get('text', 'Résultat sans texte')) 
                        for result in results
                    ])
                    return formatted_results
                
                # Si pas de résultats, mais autres informations
                if 'question' in response:
                    return str(response['question'])
                
                # Cas par défaut pour les dictionnaires
                return str(response)
            
            # Conversion générique pour d'autres types de dictionnaires
            return str(response)
        
        # Pour les autres types de réponses
        return str(response)
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête : {e}")
        # Log de la trace complète de l'erreur
        import traceback
        logger.error(traceback.format_exc())
        return f"Erreur : {str(e)}"

# Exemple d'utilisation
if __name__ == "__main__":
    test_request = "Trouve et analyse les dernières tendances de l'IA, avec des sources et des données"
    process_user_request(test_request, debug_mode=True)
