import logging
from typing import Dict, Any, Optional, Union
from uuid import uuid4
import json
from phi.agent import Agent

# Importer les agents
from agents.agent_base import get_agent_base  


class WebSocketSessionManager:
    """
    Gère les sessions WebSocket avec persistance du contexte
    et routing dynamique entre agents
    """
    def __init__(self):
        # Sessions actives par user_id
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, user_id: str) -> str:
        """
        Crée une nouvelle session pour un utilisateur
        
        Args:
            user_id (str): Identifiant de l'utilisateur
        
        Returns:
            str: ID de session unique
        """
        session_id = str(uuid4())
        
        # Session initiale avec AgentBase (au lieu de UserProxy)
        self.active_sessions[user_id] = {
            'session_id': session_id,
            'current_agent': 'agent_base',  # Changement ici
            'conversation_history': [],
            'context': {}
        }
        
        return session_id
    
    def get_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère la session d'un utilisateur
        
        Args:
            user_id (str): Identifiant de l'utilisateur
        
        Returns:
            Optional[Dict[str, Any]]: Session de l'utilisateur ou None
        """
        return self.active_sessions.get(user_id)
    
    def switch_agent(self, user_id: str, new_agent: str) -> bool:
        """
        Bascule vers un nouvel agent pour une session
        
        Args:
            user_id (str): Identifiant de l'utilisateur
            new_agent (str): Nom du nouvel agent
        
        Returns:
            bool: True si le changement est réussi, False sinon
        """
        session = self.get_session(user_id)
        if not session:
            return False
        
        # Liste des agents disponibles
        available_agents = {
            'agent_base': get_agent_base,  # Nouvel agent ajouté
        }
        
        if new_agent not in available_agents:
            logger.warning(f"Agent {new_agent} non disponible")
            return False
        
        session['current_agent'] = new_agent
        return True
    
    def add_to_conversation_history(
        self, 
        user_id: str, 
        message: str, 
        role: str = 'user'
    ):
        """
        Ajoute un message à l'historique de conversation
        
        Args:
            user_id (str): Identifiant de l'utilisateur
            message (str): Message à ajouter
            role (str): Rôle du message (user/assistant)
        """
        session = self.get_session(user_id)
        if session:
            session['conversation_history'].append({
                'role': role,
                'content': message
            })
        print('>>>>>>>>>>>>>>>')
        print(session['conversation_history'])

    def get_current_agent(self, user_id: str) -> Union[Dict[str, Any], None]:
        """
        Récupère l'agent courant pour une session avec sa configuration de widget
        
        Args:
            user_id (str): Identifiant de l'utilisateur
        
        Returns:
            Dict[str, Any] ou None: Dictionnaire contenant l'agent et les widgets
        """
        # Vérifier que la session existe
        if user_id not in self.active_sessions:
            logging.warning(f"Aucune session trouvée pour {user_id}")
            # Créer une nouvelle session si elle n'existe pas
            self.create_session(user_id)
        
        session = self.active_sessions[user_id]
        
        # Log de débogage
        logging.info(f"Récupération de l'agent pour {user_id}")
        
        # Récupération de l'agent de base
        agent_response = get_agent_base(
            user_id=user_id, 
            session_id=session.get('session_id')
        )
        
        # Log du nombre de widgets
        if 'widget_list' in agent_response:
            logging.info(f"Nombre de widgets générés : {len(agent_response['widget_list'])}")
        
        return agent_response

    def connect(self, websocket, user_id: str):
        """
        Enregistre une connexion WebSocket active pour un utilisateur
        
        Args:
            websocket: Connexion WebSocket active
            user_id (str): Identifiant de l'utilisateur
        """
        if user_id not in self.active_sessions:
            self.create_session(user_id)
        
        # Stocker la connexion WebSocket dans la session
        self.active_sessions[user_id]['websocket'] = websocket
        
        # Optionnel : Ajouter des logs
        logging.info(f"Connexion WebSocket établie pour l'utilisateur {user_id}")

# Instance globale du gestionnaire de session
websocket_session_manager = WebSocketSessionManager()
