import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional, Dict, Any

from api.models.websocket import WebSocketRequest, WebSocketResponse
from api.websocket.connection_manager import websocket_manager
from api.websocket.session_manager import websocket_session_manager

# Configuration du logger
logger = logging.getLogger(__name__)

# Création du routeur
websocket_router = APIRouter()

@websocket_router.websocket("")  # Route vide car le préfixe /ws est déjà ajouté par v1_router

async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = Query(..., description="ID de l'utilisateur")
):
    """
    Endpoint WebSocket principal pour les interactions avec les agents Phidata
    """
    logger.info(f"Nouvelle tentative de connexion WebSocket - user_id: {user_id}")
    
    try:
        # Initialiser la session et la connexion
        session_id = websocket_session_manager.create_session(user_id)
        await websocket_manager.connect(websocket, user_id)
        
        # Message de bienvenue
        welcome_response = WebSocketResponse(
            status="success",
            message="Connexion établie",
            data={"session_id": session_id}
        )
        await websocket.send_text(welcome_response.model_dump_json())
        logger.info(f"Message de bienvenue envoyé - user_id: {user_id}")
        
        # Boucle de réception des messages
        while True:
            try:
                # Réception et validation du message
                data = await websocket.receive_text()
                logger.debug(f"Message reçu de {user_id}: {data}")
                
                try:
                    message_data = json.loads(data)
                    # Ajouter user_id au message s'il n'est pas présent
                    if isinstance(message_data, dict) and "user_id" not in message_data:
                        message_data["user_id"] = user_id
                    request = WebSocketRequest(**message_data)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Format de message invalide de {user_id}: {str(e)}")
                    await websocket.send_text(
                        WebSocketResponse(
                            status="error",
                            message="Format de message invalide",
                            data={"error": str(e)}
                        ).model_dump_json()
                    )
                    continue
                
                # Récupérer et valider l'agent
                current_agent_response = websocket_session_manager.get_current_agent(user_id)

                # Extraction de l'agent et de la configuration du widget
                current_agent = current_agent_response['agent']
                widget_config = current_agent_response.get('widget_config', {
                    'name': 'default_select',
                    'type': 'select',
                    'options': ['Option par défaut 1', 'Option par défaut 2']
                })

                if not current_agent:
                    logger.warning(f"Aucun agent disponible pour {user_id}")
                    await websocket.send_text(
                        WebSocketResponse(
                            status="error",
                            message="Aucun agent disponible",
                            data={"session_id": session_id}
                        ).model_dump_json()
                    )
                    continue

                # Exécution de l'agent
                try:
                    response = current_agent.run(request.query)
                    
                    # Extraction du contenu JSON de la réponse
                    if hasattr(response, 'content'):
                        try:
                            # Tenter de parser le contenu comme JSON
                            parsed_response = json.loads(response.content)
                        except (json.JSONDecodeError, TypeError):
                            # Si le parsing échoue, utiliser le contenu tel quel
                            parsed_response = {"status": "error", "content": str(response.content)}
                    elif isinstance(response, str):
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            parsed_response = {"status": "error", "content": response}
                    else:
                        # Convertir l'objet en dictionnaire si possible
                        parsed_response = {"status": "error", "content": str(response)}

                    logging.info(f"🌈 Configuration du widget : {widget_config}")
                    
                    # Préparation de la réponse de l'agent
                    ws_response = WebSocketResponse(
                        status="success",
                        message="Réponse de l'agent",
                        data=parsed_response
                    )
                    await websocket.send_text(ws_response.model_dump_json())
                    
                    # Envoi de la configuration du widget
                    widget_response = WebSocketResponse(
                        status="dynamic_widget",
                        message="Configuration de widget",
                        data=widget_config
                    )
                    logging.info(f"🚀 Configuration du widget à envoyer : {widget_config}")
                    logging.info(f"📡 Envoi du widget : {widget_response}")
                    await websocket.send_text(widget_response.model_dump_json())

                except Exception as e:
                    logger.error(f"Erreur d'exécution de l'agent : {e}")
                    await websocket.send_text(
                        WebSocketResponse(
                            status="error",
                            message="Erreur lors de l'exécution de l'agent",
                            data={"error": str(e)}
                        ).model_dump_json()
                    )
                
                # Mettre à jour l'historique
                websocket_session_manager.add_to_conversation_history(
                    user_id, request.query, 'user'
                )
                websocket_session_manager.add_to_conversation_history(
                    user_id, str(response), 'assistant'
                )
                
            except WebSocketDisconnect:
                logger.info(f"Client déconnecté: {user_id}")
                break
            
            except Exception as e:
                logger.error(f"Erreur inattendue pour {user_id}: {str(e)}")
                try:
                    await websocket.send_text(
                        WebSocketResponse(
                            status="error",
                            message="Erreur inattendue",
                            data={"error": str(e)}
                        ).model_dump_json()
                    )
                except:
                    break
    
    except Exception as e:
        logger.error(f"Erreur critique WebSocket pour {user_id}: {str(e)}")
    
    finally:
        # Nettoyage
        try:
            websocket_manager.disconnect(websocket, user_id)
            logger.info(f"Nettoyage effectué pour {user_id}")
        except Exception as cleanup_error:
            logger.error(f"Erreur lors du nettoyage pour {user_id}: {str(cleanup_error)}")
