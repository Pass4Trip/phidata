from typing import Optional, Dict, Any, List, Callable, Union
from phi.agent import Agent
from phi.llm.openai import OpenAIChat
import logging
import queue
import threading
import uuid
import json
from datetime import datetime
import os
from pika.spec import Basic, BasicProperties

# Importer pika correctement
import pika
from pika.adapters.blocking_connection import BlockingChannel
from dotenv import load_dotenv
import traceback
import sys

# Charger les variables d'environnement
load_dotenv()

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration du logging
logger = logging.getLogger(__name__)

def default_chatbot_message_sender(message):
    """
    Sender de messages par défaut qui log les messages
    
    Args:
        message (dict): Message à envoyer
    """
    logger.info(f"MESSAGE PROACTIF PAR DÉFAUT : {message}")

class UserProxyAgent:
    def __init__(
        self, 
        model: Optional[str] = None,
        debug_mode: bool = False,
        rabbitmq_config: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Agent proxy pour la gestion des interactions utilisateur.
        
        Args:
            model (str, optional): Modèle LLM à utiliser
            debug_mode (bool): Mode débogage
            rabbitmq_config (dict, optional): Configuration de connexion RabbitMQ
        """
        # Configuration du logging
        logging.basicConfig(
            level=logging.DEBUG if debug_mode else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialiser le logger
        self.logger = logging.getLogger(__name__)
        
        # Configurer un sender par défaut
        self._chatbot_message_sender = UserProxyAgent.get_default_message_sender()
        
        # Configuration RabbitMQ par défaut depuis les variables d'environnement
        default_rabbitmq_config = {
            'host': os.getenv('RABBITMQ_HOST', 'localhost'),
            'port': int(os.getenv('RABBITMQ_PORT', 30645)),
            'user': os.getenv('RABBITMQ_USER', 'guest'),
            'password': os.getenv('RABBITMQ_PASSWORD', 'guest'),
            'queue_clarification': os.getenv('QUEUE_CLARIFICATION', 'queue_clarification'),
            'queue_progress_task': os.getenv('QUEUE_PROGRESS_TASK', 'queue_progress_task')
        }

        # Fusionner la configuration personnalisée avec la configuration par défaut
        self.rabbitmq_config = {**default_rabbitmq_config, **(rabbitmq_config or {})}

        # Initialisation de l'agent
        self.agent = Agent(
            name="UserProxy",
            llm=OpenAIChat(
                model=model or "gpt-4o-mini",
                temperature=0.3
            ),
            instructions=[
                "Tu es un agent proxy qui gère les interactions utilisateur.",
                "Tes responsabilités principales sont :",
                "1. Être le point de contact unique pour l'utilisateur",
                "2. Analyser et router les demandes vers l'agent approprié",
                "3. Gérer les demandes de clarification",
                "4. Suivre et communiquer l'avancement des traitements",
                "5. Communiquer de manière claire et professionnelle"
            ],
            debug_mode=debug_mode
        )
        
        # Files d'attente pour les messages
        self._clarification_queue = queue.Queue(maxsize=1000)
        self._progress_queue = queue.Queue(maxsize=1000)
        
        # Événement pour contrôler les threads
        self._stop_event = threading.Event()
        
        # Dictionnaire pour suivre les conversations
        self._active_conversations = {}
        
        # Dictionnaire pour suivre les tâches complexes
        self._ongoing_tasks = {}
        
        # Threads de traitement
        self._clarification_thread = None
        self._progress_thread = None
        
        # Configuration RabbitMQ
        self._rabbitmq_connection = None
        self._rabbitmq_channel = None

    def _connect_to_rabbitmq(self, create_channel: bool = True):
        """
        Établit une connexion à RabbitMQ et optionnellement configure le canal.

        Args:
            create_channel (bool): Si True, crée un canal RabbitMQ. 
                                   Si False, retourne uniquement la connexion.

        Returns:
            Union[pika.BlockingConnection, Tuple[pika.BlockingConnection, pika.channel.Channel]]: 
            Connexion RabbitMQ, et optionnellement le canal
        """
        try:
            # Paramètres de connexion
            connection_params = pika.ConnectionParameters(
                host=self.rabbitmq_config.get('host', 'localhost'),
                port=self.rabbitmq_config.get('port', 5672),
                credentials=pika.PlainCredentials(
                    username=self.rabbitmq_config.get('username', 'guest'),
                    password=self.rabbitmq_config.get('password', 'guest')
                )
            )
            
            # Établir la connexion
            connection = pika.BlockingConnection(connection_params)
            
            if create_channel:
                # Créer et stocker le canal
                channel = connection.channel()
                self._rabbitmq_connection = connection
                self._rabbitmq_channel = channel
                
                logger.info(" Connexion RabbitMQ établie avec succès")
                return connection, channel
            
            return connection

        except Exception as e:
            logger.error(f"Erreur de connexion RabbitMQ : {e}")
            raise

    # Conserver pour compatibilité, mais marquer comme deprecated
    def _get_rabbitmq_connection(self):
        """
        DEPRECATED: Utilisez _connect_to_rabbitmq(create_channel=False) à la place.
        
        Établit une connexion à RabbitMQ.
        
        Returns:
            pika.BlockingConnection: Connexion RabbitMQ
        """
        logger.warning(
            "La méthode _get_rabbitmq_connection() est obsolète. "
            "Utilisez _connect_to_rabbitmq(create_channel=False)"
        )
        return self._connect_to_rabbitmq(create_channel=False)

    def process_rabbitmq_queue(
        self, 
        queue_type: str = 'queue_clarification',
        max_retries: int = 3,
        retry_delay: int = 5,
        prefetch_count: int = 1
    ):
        """
        Traite de manière continue les messages d'une queue RabbitMQ.

        Cette méthode générique gère le traitement des messages pour différents types de queues,
        avec une logique de reconnexion et de gestion des erreurs.

        Args:
            queue_type (str): Type de queue à traiter
                - 'queue_clarification' : Demandes de clarification
                - 'queue_progress_task' : Progression des tâches
            max_retries (int): Nombre maximum de tentatives de reconnexion
            retry_delay (int): Délai entre les tentatives de reconnexion
            prefetch_count (int): Nombre de messages à précharger simultanément

        Raises:
            ValueError: Si le type de queue est invalide
        """
        # Validation du type de queue
        if queue_type not in ['queue_clarification', 'queue_progress_task']:
            raise ValueError(f"Type de queue invalide : {queue_type}")

        # Configuration du logger spécifique
        logger_queue = logging.getLogger(f'{queue_type}_listener')
        
        # Variables de contrôle
        retry_count = 0
        should_stop = threading.Event()

        def on_connection_error(connection, exception):
            """
            Gère les erreurs de connexion avec une stratégie de backoff exponentiel
            """
            nonlocal retry_count
            retry_count += 1
            
            if retry_count <= max_retries:
                wait_time = retry_delay * (2 ** retry_count)  # Backoff exponentiel
                logger_queue.warning(
                    f"Erreur de connexion {queue_type} (Tentative {retry_count}/{max_retries}). "
                    f"Nouvelle tentative dans {wait_time} secondes"
                )
                time.sleep(wait_time)
                return True  # Autoriser une nouvelle tentative
            
            logger_queue.error(f"Nombre max de tentatives atteint pour {queue_type}")
            should_stop.set()
            return False

        def message_callback(channel, method, properties, body):
            """
            Traitement standard des messages avec gestion des erreurs
            """
            try:
                # Sélection dynamique du processeur
                message_processor = (
                    self._process_clarification_message 
                    if queue_type == 'queue_clarification' 
                    else self._process_progress_message
                )
                
                # Traitement du message
                message_processor(channel, method, properties, body)
                
                # Acquittement du message
                channel.basic_ack(delivery_tag=method.delivery_tag)
                
                # Réinitialiser le compteur de tentatives
                nonlocal retry_count
                retry_count = 0

            except Exception as e:
                logger_queue.error(f"Erreur de traitement {queue_type}: {e}")
                # Réessayer ou rejeter le message
                channel.basic_nack(
                    delivery_tag=method.delivery_tag, 
                    requeue=True  # Renvoyer le message dans la queue
                )

        while not should_stop.is_set():
            try:
                # Établissement de la connexion
                connection, channel = self._connect_to_rabbitmq()
                
                # Configuration de la queue
                queue_name = self.rabbitmq_config.get(queue_type, queue_type)
                channel.queue_declare(queue=queue_name, durable=True)
                
                # Configuration de la consommation
                channel.basic_qos(prefetch_count=prefetch_count)
                channel.basic_consume(
                    queue=queue_name, 
                    on_message_callback=message_callback
                )
                
                logger_queue.info(f" Démarrage de l'écoute continue sur {queue_name}")
                
                # Boucle de consommation
                try:
                    # Utilisation d'un thread séparé pour start_consuming()
                    def consume_messages():
                        try:
                            channel.start_consuming()
                        except (KeyboardInterrupt, SystemExit):
                            logger_queue.info(f"Interruption de la consommation pour {queue_name}")
                            channel.stop_consuming()
                        except Exception as consume_error:
                            logger_queue.error(f"Erreur lors de la consommation de {queue_name}: {consume_error}")
                        finally:
                            should_stop.set()

                    # Création et démarrage du thread de consommation
                    consume_thread = threading.Thread(
                        target=consume_messages, 
                        name=f"{queue_name}_ConsumerThread",
                        daemon=True
                    )
                    consume_thread.start()

                    # Attente avec possibilité d'interruption
                    while not should_stop.is_set():
                        should_stop.wait(timeout=1)  # Vérification périodique
                        
                    # Arrêt propre du thread de consommation
                    if consume_thread.is_alive():
                        channel.stop_consuming()
                        consume_thread.join(timeout=5)

                except Exception as main_error:
                    logger_queue.critical(f"Erreur critique dans la consommation de {queue_name}: {main_error}")
                    should_stop.set()

            except (pika.exceptions.AMQPConnectionError, 
                    pika.exceptions.AMQPChannelError) as conn_error:
                
                if not on_connection_error(connection, conn_error):
                    break

            except Exception as unexpected_error:
                logger_queue.critical(
                    f"Erreur inattendue dans {queue_type}: {unexpected_error}"
                )
                should_stop.set()

            # Pause entre les tentatives
            time.sleep(retry_delay)

        logger_queue.info(f"Arrêt de l'écoute pour {queue_type}")

    def start_processing(
        self, 
        queue_type: str = 'queue_clarification',
        max_retries: int = 3, 
        retry_delay: int = 5,
        prefetch_count: int = 1
    ):
        """
        Démarre le traitement des messages dans un thread séparé.

        Args:
            queue_type (str): Type de queue à traiter
            max_retries (int): Nombre maximum de tentatives
            retry_delay (int): Délai entre les tentatives
            prefetch_count (int): Nombre de messages à précharger
        
        Returns:
            threading.Thread: Thread de traitement des messages
        """
        if queue_type not in ['queue_clarification', 'queue_progress_task']:
            raise ValueError(f"Type de queue invalide. Doit être 'queue_clarification' ou 'queue_progress_task'")

        # Création du thread pour l'écoute continue
        thread = threading.Thread(
            target=self.process_rabbitmq_queue, 
            kwargs={
                'queue_type': queue_type, 
                'max_retries': max_retries, 
                'retry_delay': retry_delay,
                'prefetch_count': prefetch_count
            },
            name=f"{queue_type.capitalize()}MessageProcessor",
            daemon=True  # Thread en arrière-plan
        )
        thread.start()
        
        # Stocker la référence du thread avec des attributs dynamiques
        thread_attr = f'_{queue_type}_thread'
        setattr(self, thread_attr, thread)
        
        return thread

    def start_processing_tasks(self):
        """
        Démarre le traitement des files de clarification et de progression.
        """
        self.start_processing(queue_type='queue_clarification')
        self.start_processing(queue_type='queue_progress_task')
    
    def enqueue_clarification_request(
        self, 
        request: Dict[str, Any], 
        queue_type: str = 'queue_clarification'
    ) -> Optional[str]:
        """
        Ajoute une demande de clarification à la file d'attente.
        
        Args:
            request (dict): Demande de clarification
            queue_type (str): Type de queue
        
        Returns:
            str: ID de la conversation
        """
        try:
            # Générer un ID de conversation
            conversation_id = str(uuid.uuid4())
            request['conversation_id'] = conversation_id
            
            # Sélection de la queue appropriée
            current_queue = (
                self._clarification_queue if queue_type == 'queue_clarification'
                else self._progress_queue
            )
            
            # Ajouter à la file d'attente
            current_queue.put(request, block=False)
            
            logger.info(f"Demande de {queue_type} ajoutée à la file")
            return conversation_id
        
        except queue.Full:
            logger.error("La file d'attente est pleine")
            return None
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout à la file : {e}")
            return None
    
    
    
    def stop_processing(self):
        """
        Arrête proprement le traitement des demandes RabbitMQ.
        
        Méthode sécurisée pour :
        - Signaler l'arrêt aux threads
        - Libérer les ressources
        - Gérer les exceptions potentielles
        """
        try:
            # Threads à arrêter
            threads_to_stop = [
                getattr(self, '_clarification_thread', None),
                getattr(self, '_progress_thread', None)
            ]
            
            # Compteur de threads arrêtés
            stopped_threads = 0
            
            for thread in threads_to_stop:
                if thread and thread.is_alive():
                    try:
                        # Utiliser un mécanisme de signalement
                        if hasattr(thread, 'stop_event'):
                            thread.stop_event.set()
                        
                        # Attente avec timeout
                        thread.join(timeout=5)
                        
                        if not thread.is_alive():
                            stopped_threads += 1
                            logger.info(f"Thread {thread.name} arrêté correctement")
                        else:
                            logger.warning(f"Thread {thread.name} n'a pas pu être arrêté")
                    
                    except Exception as thread_error:
                        logger.error(f"Erreur lors de l'arrêt d'un thread : {thread_error}")
            
            # Log du résultat global
            if stopped_threads > 0:
                logger.info(f"{stopped_threads} thread(s) de traitement RabbitMQ arrêté(s)")
            else:
                logger.info("Aucun thread de traitement actif")
        
        except Exception as global_error:
            logger.critical(f"Erreur critique lors de l'arrêt des processus : {global_error}")
        
        finally:
            # Réinitialisation des références de threads
            self._clarification_thread = None
            self._progress_thread = None
    
    async def _process_clarification_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un message de demande de clarification.
        
        Args:
            message (dict): Message de clarification
        
        Returns:
            dict: Réponse de clarification
        """
        try:
            # Extraire les informations nécessaires
            original_request = message.get('original_request', '')
            clarification_details = message.get('clarification_details', '')
            source_agent = message.get('source_agent', 'unknown')
            
            # Générer la demande de clarification via le LLM
            clarification_prompt = (
                f"Une clarification est nécessaire pour la requête : '{original_request}'\n"
                f"Détails de clarification requis : {clarification_details}\n"
                "Veuillez fournir des informations supplémentaires."
            )
            
            # Obtenir la réponse de clarification
            clarification_response = self.agent.run(clarification_prompt)
            
            # Préparer la réponse à renvoyer
            response_message = {
                'status': 'clarification_received',
                'original_request': original_request,
                'clarification_response': clarification_response,
                'source_agent': source_agent
            }
            
            return response_message
        
        except Exception as e:
            logger.error(f"Erreur de traitement de la clarification : {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _validate_message_sender(self):
        """
        Valide et configure le sender de messages.
        
        Raises:
            ValueError: Si aucun sender valide n'est trouvé
        """
        # Vérifier l'existence du sender
        if not hasattr(self, '_chatbot_message_sender'):
            logger.warning("Aucun sender de message chatbot configuré. Utilisation du sender par défaut.")
            self._chatbot_message_sender = self.get_default_message_sender()
        
        # Vérifier si le sender est callable
        if not callable(self._chatbot_message_sender):
            logger.error(
                f"Le sender de message n'est pas callable. "
                f"Type actuel : {type(self._chatbot_message_sender)}"
            )
            self._chatbot_message_sender = self.get_default_message_sender()
        
        # Validation finale
        if not callable(self._chatbot_message_sender):
            raise ValueError("Impossible de configurer un sender de message valide")

    def _process_progress_message(
        self, 
        ch, 
        method: pika.spec.Basic.Deliver, 
        properties: pika.spec.BasicProperties, 
        body: bytes
    ):
        """
        Traite un message de progression RabbitMQ.
        
        Args:
            ch: Canal RabbitMQ
            method: Méthode de livraison
            properties: Propriétés du message
            body: Corps du message
        """
        try:
            # Convertir le message en dictionnaire
            message = json.loads(body.decode('utf-8'))
            
            # Validation du sender de messages
            self._validate_message_sender()
            
            # Envoi du message via le sender
            try:
                self._chatbot_message_sender(message)
                logger.info("Message envoyé avec succès via le sender")
            except Exception as send_error:
                logger.error(f"Erreur lors de l'envoi du message : {send_error}")
                logger.debug(f"Détails du message : {message}")
                # Ne pas renvoyer le message, mais logger l'erreur
            
            # Acquitter le message
            ch.basic_ack(delivery_tag=method.delivery_tag)
        
        except json.JSONDecodeError:
            logger.error("Erreur de décodage JSON du message")
            # Ne pas acquitter le message en cas d'erreur de décodage
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message : {e}")
            # Ne pas acquitter le message en cas d'erreur
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    def start_consuming(self):
        """
        DEPRECATED: Utilisez process_rabbitmq_queue() à la place.
        
        Cette méthode est conservée pour des raisons de compatibilité 
        mais ne devrait plus être utilisée.
        """
        logger.warning(
            "La méthode start_consuming() est obsolète. "
            "Utilisez process_rabbitmq_queue() avec le type de queue approprié."
        )
        try:
            self.process_rabbitmq_queue()
        except Exception as e:
            logger.error(f"Erreur lors de la consommation des messages : {e}")
            raise
    
    def _translate_progress_to_conversation(self, message):
        """
        Traduire un message de progression technique en message conversationnel
        
        Args:
            message (dict): Message de progression à traduire
        
        Returns:
            str: Message conversationnel
        """
        task_type_translations = {
            'calculation': "Calcul en cours",
            'web_search': "Recherche sur le web",
            'synthesis': "Synthèse finale",
            'subtask': "Étape intermédiaire",
            'task': "Tâche principale"
        }
        
        status_translations = {
            'started': "Je commence à travailler sur votre demande...",
            'in_progress': "Je suis en train de traiter cette étape...",
            'completed': "Cette étape est terminée !",
            'pending': "Préparation de l'étape...",
            'error': "Oups, j'ai rencontré un problème."
        }
        
        # Traduction du type de tâche
        task_type_fr = task_type_translations.get(
            message.get('task_type', 'subtask'), 
            "Étape du processus"
        )
        
        # Traduction du statut
        status_fr = status_translations.get(
            message.get('status', 'in_progress'), 
            "en cours"
        )
        
        # Construction du message conversationnel
        conversation_message = f" {task_type_fr} : {status_fr}\n"
        
        # Ajouter des détails si disponibles
        if 'subtasks' in message:
            subtasks_info = [
                f"- {subtask.get('description', 'Sous-tâche')} : {subtask.get('status', 'en attente')}"
                for subtask in message.get('subtasks', [])
            ]
            if subtasks_info:
                conversation_message += "Détails des sous-tâches :\n" + "\n".join(subtasks_info)
        
        result = message.get('result', {})
        if result and isinstance(result, dict):
            content = result.get('content', '')
            if content:
                conversation_message += f"\n Résultat : {content}"
        
        return conversation_message
    
    def send_proactive_message(self, message):
        """
        Envoyer un message proactif au chatbot
        
        Args:
            message (dict): Message à envoyer
        """
        # Log détaillé de débogage
        logger.info(f"DÉBOGAGE SENDER: Tentative d'envoi de message - {message}")
        
        # Vérification de l'attribut
        if not hasattr(self, '_chatbot_message_sender'):
            logger.warning("DÉBOGAGE SENDER : Attribut _chatbot_message_sender non défini")
            return
        
        # Vérification de la valeur
        if self._chatbot_message_sender is None:
            logger.warning("DÉBOGAGE SENDER : _chatbot_message_sender est None")
            return
        
        try:
            # Envoyer le message
            self._chatbot_message_sender({
                'type': 'proactive_message',
                'content': message.get('content', ''),
                'request_id': message.get('request_id'),
                'task_type': message.get('task_type'),
                'status': message.get('status'),
                'sender': 'user_proxy'
            })
            logger.info(f"DÉBOGAGE SENDER : Message envoyé avec succès - {message.get('content', '')}")
        except Exception as e:
            logger.error(f"DÉBOGAGE SENDER : Erreur lors de l'envoi du message au chatbot : {e}")
    
    def set_chatbot_message_sender(self, sender_function):
        """
        Configure le sender de messages pour le chatbot
        
        Args:
            sender_function (callable): Fonction pour envoyer des messages au chatbot
        """
        print(f" Configuration du sender de messages : {sender_function}")
        
        # Vérifier que le sender est bien callable
        if not callable(sender_function):
            print(" ERREUR : Le sender doit être une fonction callable")
            return
        
        # Configurer le sender
        self._chatbot_message_sender = sender_function
        print(" Sender de messages configuré avec succès")

    @staticmethod
    def get_default_message_sender():
        """
        Sender de message par défaut qui imprime et signale le problème
        
        Returns:
            callable: Fonction de sender par défaut
        """
        def default_sender(message):
            print(" SENDER PAR DÉFAUT UTILISÉ :")
            print("Type de message :", message.get('message_type', 'Inconnu'))
            print("Requête originale :", message.get('metadata', {}).get('original_request', 'N/A'))
            print(" Le sender Streamlit n'est PAS configuré correctement !")
        return default_sender

    def _publish_response(
        self, 
        queue_name: str, 
        message: Dict[str, Any]
    ):
        """
        Publie une réponse dans une queue RabbitMQ.
        
        Args:
            queue_name (str): Nom de la queue de destination
            message (dict): Message à publier
        """
        try:
            connection, channel = self._connect_to_rabbitmq()
            
            # Déclarer la queue
            channel.queue_declare(queue=queue_name, durable=True)
            
            # Publier le message
            channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(message).encode('utf-8')
            )
            
            logger.info(f"Message publié dans la queue {queue_name}")
            
            # Fermer la connexion
            connection.close()
        
        except Exception as e:
            logger.error(f"Erreur lors de la publication du message : {e}")



    def route_request(
        self, 
        request: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route une requête utilisateur vers l'agent le plus approprié.
        
        Args:
            request (str): Requête utilisateur à router
            context (dict, optional): Contexte supplémentaire pour le routage
        
        Returns:
            dict: Résultat du routage avec l'agent cible et autres informations
        """
        try:
            # Liste des agents disponibles avec leurs descriptions
            available_agents = {
                "web_searcher": "Agent spécialisé dans la recherche web et l'agrégation d'informations",
                "travel_planner": "Agent pour la planification et la recherche de voyages",
                "api_knowledge": "Agent de recherche et d'analyse de connaissances spécialisées",
                "data_analysis": "Agent pour l'analyse et le traitement de données",
                "orchestrator": "Agent généraliste pour les tâches complexes ou non spécifiées"
            }
            
            # Préparation du prompt de routage
            routing_prompt = f"""
            Étant donné la requête utilisateur suivante : '{request}'
            
            Sélectionne l'agent le plus approprié parmi ces options :
            {', '.join(available_agents.keys())}
            
            Règles de sélection :
            1. Analyse précisément le besoin de l'utilisateur
            2. Choisis l'agent avec la spécialisation la plus proche
            3. En cas de doute, sélectionne 'orchestrator'
            4. Retourne UNIQUEMENT le nom de l'agent, sans aucun texte supplémentaire
            
            Contexte supplémentaire disponible : {context or 'Aucun'}
            
            Agent cible :"""
            
            # Utilisation du LLM pour déterminer l'agent
            target_agent = self.agent.run(routing_prompt).strip().lower()
            
            # Validation du résultat
            if target_agent not in available_agents:
                logger.warning(f"Agent non reconnu : {target_agent}. Utilisation de l'orchestrateur.")
                target_agent = "orchestrator"
            
            # Préparation de la réponse de routage
            routing_response = {
                "status": "success",
                "target_agent": target_agent,
                "original_request": request,
                "agent_description": available_agents.get(target_agent, "Agent générique"),
                "context": context
            }
            
            # Journalisation du routage
            logger.info(f"Requête routée vers l'agent : {target_agent}")
            
            return routing_response
        
        except Exception as e:
            logger.error(f"Erreur lors du routage de la requête : {e}")
            return {
                "status": "error",
                "message": str(e),
                "original_request": request
            }

def get_user_proxy_agent(
    model: Optional[str] = None, 
    debug_mode: bool = False,
    rabbitmq_config: Optional[Dict[str, str]] = None
) -> UserProxyAgent:
    """
    Fonction factory pour créer l'agent UserProxy.
    
    Args:
        model (str, optional): Modèle LLM à utiliser
        debug_mode (bool): Mode débogage
        rabbitmq_config (dict, optional): Configuration de connexion RabbitMQ
    
    Returns:
        UserProxyAgent: Instance de l'agent UserProxy
    """
    agent = UserProxyAgent(
        model=model, 
        debug_mode=debug_mode, 
        rabbitmq_config=rabbitmq_config
    )
    
    # Démarrage automatique du traitement
    agent.start_processing_tasks()
    
    return agent
