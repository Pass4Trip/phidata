import asyncio
import websockets
import json
import logging
import urllib.parse
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import WebSocketException

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_websocket():
    """
    Test de connexion WebSocket basique
    """
    # Configuration de la connexion
    base_uri = "ws://localhost:8001/v1/ws"
    params = urllib.parse.urlencode({"user_id": "test_user"})
    uri = f"{base_uri}?{params}"
    
    logger.info(f"Tentative de connexion à {uri}")
    
    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            subprotocols=["websocket"]
        ) as ws:
            logger.info("Connexion établie")
            
            try:
                # Attendre le message de bienvenue
                response = await ws.recv()
                logger.info(f"Message de bienvenue reçu: {response}")
                
                # Envoyer un message de test
                test_message = {
                    "query": "Test message",
                    "user_id": "test_user",
                    "model_id": "gpt-4o-mini"
                }
                await ws.send(json.dumps(test_message))
                logger.info("Message de test envoyé")
                
                # Recevoir la réponse
                response = await ws.recv()
                response_data = json.loads(response)
                logger.info(f"Réponse reçue: {response_data}")
                
                # Attendre un peu pour voir si la connexion reste stable
                await asyncio.sleep(2)
                
            except json.JSONDecodeError as e:
                logger.error(f"Erreur de décodage JSON: {e}")
            except asyncio.TimeoutError:
                logger.error("Timeout lors de l'attente de la réponse")
            except Exception as e:
                logger.error(f"Erreur lors de la communication: {str(e)}")
                
    except WebSocketException as e:
        logger.error(f"Erreur WebSocket: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
