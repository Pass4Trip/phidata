from typing import List, Dict
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
import logging
import asyncio
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Gestionnaire de connexions WebSocket pour Phidata
    """
    def __init__(self):
        self.active_connections: Dict[str, List[dict]] = {}
        self.ping_interval = 20  # secondes
        self._background_tasks = set()

    async def connect(self, websocket: WebSocket, user_id: str):
        """
        Établit une nouvelle connexion WebSocket pour un utilisateur
        """
        try:
            await websocket.accept()
            
            # Créer la liste pour l'utilisateur si elle n'existe pas
            if user_id not in self.active_connections:
                self.active_connections[user_id] = []
            
            # Ajouter la connexion avec des métadonnées
            connection_info = {
                "websocket": websocket,
                "connected_at": datetime.now(),
                "last_activity": datetime.now()
            }
            self.active_connections[user_id].append(connection_info)
            
            # Démarrer la tâche de ping pour cette connexion
            task = asyncio.create_task(self._keep_alive(websocket, user_id))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            
            logger.info(f"Nouvelle connexion établie pour user_id: {user_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la connexion pour user_id {user_id}: {str(e)}")
            raise

    def disconnect(self, websocket: WebSocket, user_id: str):
        """
        Déconnecte un WebSocket spécifique pour un utilisateur
        """
        try:
            if user_id in self.active_connections:
                # Trouver et supprimer la connexion spécifique
                self.active_connections[user_id] = [
                    conn for conn in self.active_connections[user_id] 
                    if conn["websocket"] != websocket
                ]
                
                # Nettoyer si plus de connexions pour cet utilisateur
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
                    logger.info(f"Toutes les connexions fermées pour user_id: {user_id}")
                else:
                    logger.info(f"Connexion fermée pour user_id: {user_id}, {len(self.active_connections[user_id])} connexion(s) restante(s)")
        except Exception as e:
            logger.error(f"Erreur lors de la déconnexion pour user_id {user_id}: {str(e)}")

    async def send_message(self, message: str, user_id: str):
        """
        Envoie un message à toutes les connexions d'un utilisateur
        """
        if user_id in self.active_connections:
            disconnected = []
            for conn in self.active_connections[user_id]:
                try:
                    await conn["websocket"].send_text(message)
                    conn["last_activity"] = datetime.now()
                except Exception as e:
                    logger.error(f"Erreur d'envoi pour user_id {user_id}: {str(e)}")
                    disconnected.append(conn)
            
            # Nettoyer les connexions mortes
            if disconnected:
                for conn in disconnected:
                    self.disconnect(conn["websocket"], user_id)

    async def broadcast(self, message: str):
        """
        Envoie un message à tous les utilisateurs connectés
        """
        for user_id in list(self.active_connections.keys()):
            await self.send_message(message, user_id)

    async def _keep_alive(self, websocket: WebSocket, user_id: str):
        """
        Maintient la connexion WebSocket active avec des pings réguliers
        """
        try:
            while True:
                await asyncio.sleep(self.ping_interval)
                try:
                    ping_message = json.dumps({"type": "ping"})
                    await websocket.send_text(ping_message)
                except Exception as e:
                    logger.warning(f"Échec du ping pour user_id {user_id}: {str(e)}")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Erreur dans keep_alive pour user_id {user_id}: {str(e)}")
        finally:
            self.disconnect(websocket, user_id)

# Créer une instance globale du gestionnaire
websocket_manager = ConnectionManager()
