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
    logger.info(f"🌐 Nouvelle tentative de connexion WebSocket - user_id: {user_id}")
    
    try:
        # Accepter la connexion WebSocket
        await websocket.accept()
        logger.info(f"🌐 Connexion WebSocket acceptée pour {user_id}")
        
        # Initialiser la session et la connexion
        session_id = websocket_session_manager.create_session(user_id)
        logger.info(f"📡 Session initialisée pour {user_id}")
        
        # Message de bienvenue
        welcome_response = WebSocketResponse(
            status="success",
            message="Connexion établie",
            data={"session_id": session_id}
        )
        await websocket.send_text(welcome_response.model_dump_json())
        logger.info(f"📩 Message de bienvenue envoyé - user_id: {user_id}")
        
        while True:
            try:
                # Réception du message
                data = await websocket.receive_text()
                logger.info(f"📥 Message reçu de {user_id}: {data}")
                
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

                # Extraction de l'agent 
                current_agent = current_agent_response['agent']

                if not current_agent:
                    logger.warning(f"Aucun agent disponible pour {user_id}")
                    await websocket.send_text(
                        WebSocketResponse(
                            status="error",
                            message="Aucun agent disponible",
                            data={"user_id": user_id}
                        ).model_dump_json()
                    )
                    continue

                try:
                    # Traitement du message
                    response = await current_agent.arun(request.query)

                    # Convertir l'objet en dictionnaire si possible
                    parsed_response = {"status": "success", "data": {}}

                    # Gestion du contenu de la réponse
                    if isinstance(response, dict):
                        parsed_response['data'] = response
                    elif isinstance(response, str):
                        parsed_response['data']['response'] = response
                    else:
                        parsed_response['data']['response'] = str(response)

                    # Vérifier et ajouter les widgets si disponibles
                    widget_generator_result = websocket_session_manager.get_current_agent(user_id)
                    logging.info(f"🚀 Résultat du générateur de widgets : {widget_generator_result}")
                    
                    # Ajouter la liste des widgets à la réponse
                    if widget_generator_result and 'widget_list' in widget_generator_result:
                        parsed_response['data']['widget_list'] = widget_generator_result['widget_list']
                        logging.info(f"🌟 Liste des widgets à envoyer : {parsed_response['data']['widget_list']}")

                    # Préparation de la réponse de l'agent
                    ws_response = WebSocketResponse(
                        status="success",
                        message="Réponse de l'agent",
                        data=parsed_response['data']
                    )
                    await websocket.send_text(ws_response.model_dump_json())

                except Exception as e:
                    logger.error(f"❌ Erreur lors du traitement du message pour {user_id}: {str(e)}")
                    await websocket.send_text(
                        WebSocketResponse(
                            status="error",
                            message=str(e),
                            data={"user_id": user_id}
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
                logger.warning(f"🚪 Client déconnecté: {user_id}")
                break
            
            except Exception as e:
                logger.error(f"🔥 Erreur inattendue pour {user_id}: {str(e)}")
                try:
                    await websocket.send_text(
                        WebSocketResponse(
                            status="error",
                            message="Erreur inattendue",
                            data={"error": str(e)}
                        ).model_dump_json()
                    )
                except:
                    logger.error(f"❌ Impossible d'envoyer le message d'erreur à {user_id}")
                    break
    
    except Exception as e:
        logger.critical(f"🚨 Erreur critique WebSocket pour {user_id}: {str(e)}")
    
    finally:
        # Nettoyage
        try:
            websocket_manager.disconnect(websocket, user_id)
            logger.info(f"🧹 Nettoyage effectué pour {user_id}")
        except Exception as cleanup_error:
            logger.error(f"❌ Erreur lors du nettoyage pour {user_id}: {str(cleanup_error)}")
