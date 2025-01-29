from typing import Dict, Any, Optional
from uuid import uuid4
import json
import logging

# Importer les agents
from agents.user_proxy import get_user_proxy_agent
from agents.orchestrator import get_orchestrator_agent
# Ajouter ici d'autres imports d'agents si nécessaire

logger = logging.getLogger(__name__)

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
        
        # Session initiale avec UserProxy
        self.active_sessions[user_id] = {
            'session_id': session_id,
            'current_agent': 'user_proxy',
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
            'user_proxy': get_user_proxy_agent,
            'orchestrator': get_orchestrator_agent,
            # Ajouter d'autres agents ici
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
    
    def get_current_agent(self, user_id: str):
        """
        Récupère l'agent courant pour une session
        
        Args:
            user_id (str): Identifiant de l'utilisateur
        
        Returns:
            Agent: Agent courant de la session
        """
        session = self.get_session(user_id)
        if not session:
            return None
        
        agent_map = {
            'user_proxy': get_user_proxy_agent,
            'orchestrator': get_orchestrator_agent,
            # Autres agents...
        }
        
        agent_func = agent_map.get(session['current_agent'])
        if agent_func:
            return agent_func(
                user_id=user_id, 
                session_id=session['session_id'],
                conversation_history=session['conversation_history']
            )
        
        return None

# Instance globale du gestionnaire de session
websocket_session_manager = WebSocketSessionManager()
