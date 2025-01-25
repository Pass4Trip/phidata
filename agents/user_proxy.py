from typing import Optional, Dict, Any, List
from phi.agent import Agent
from phi.llm.openai import OpenAIChat
from phi.tools import WebSearch
from phi.tools.python import PythonTools
import logging
import queue
import threading
import uuid
import pika
import json
from datetime import datetime

# Configuration du logging
logger = logging.getLogger(__name__)

class UserProxyAgent:
    def __init__(
        self, 
        model: Optional[str] = None,
        debug_mode: bool = False,
        rabbitmq_config: Optional[Dict[str, str]] = None
    ):
        """
        Agent proxy pour la gestion des interactions utilisateur.
        
        Args:
            model (str, optional): Modèle LLM à utiliser
            debug_mode (bool): Mode débogage
            rabbitmq_config (dict, optional): Configuration de connexion RabbitMQ
        """
        # Configuration RabbitMQ par défaut
        self.rabbitmq_config = rabbitmq_config or {
            'host': 'localhost',
            'port': 5672,
            'clarification_queue': 'user_proxy_clarification',
            'progress_queue': 'user_proxy_progress'
        }
        
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
            tools=[
                WebSearch(),
                PythonTools()
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
    
    def _get_rabbitmq_connection(self):
        """
        Établit une connexion RabbitMQ.
        
        Returns:
            pika.BlockingConnection: Connexion RabbitMQ
        """
        try:
            connection_params = pika.ConnectionParameters(
                host=self.rabbitmq_config['host'],
                port=self.rabbitmq_config.get('port', 5672)
            )
            return pika.BlockingConnection(connection_params)
        except Exception as e:
            logger.error(f"Erreur de connexion RabbitMQ : {e}")
            raise
    
    def handle_clarification_request(
        self, 
        queue_type: str = 'clarification'
    ):
        """
        Gère les demandes de clarification de manière synchrone.
        
        Args:
            queue_type (str): Type de queue à écouter 
            ('clarification' ou 'progress')
        """
        def process_messages():
            """
            Traitement continu des messages.
            """
            while not self._stop_event.is_set():
                try:
                    # Sélection de la queue
                    current_queue = (
                        self._clarification_queue if queue_type == 'clarification'
                        else self._progress_queue
                    )
                    
                    # Récupérer un message avec un timeout
                    try:
                        message = current_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    
                    # Sélection du gestionnaire de messages
                    message_handler = (
                        self._process_clarification_message if queue_type == 'clarification'
                        else self._process_progress_message
                    )
                    
                    # Traitement du message
                    conversation_id = message.get('conversation_id', str(uuid.uuid4()))
                    
                    try:
                        # Traiter le message
                        response = message_handler(message)
                        
                        # Stocker la conversation
                        self._active_conversations[conversation_id] = {
                            'request': message,
                            'response': response,
                            'timestamp': threading.get_ident()
                        }
                        
                        # Journalisation
                        logger.info(f"Traitement {queue_type} pour la conversation {conversation_id}")
                    
                    except Exception as e:
                        logger.error(f"Erreur de traitement du message {conversation_id}: {e}")
                    
                    # Marquer la tâche comme terminée
                    current_queue.task_done()
                
                except Exception as e:
                    logger.error(f"Erreur dans le traitement des messages {queue_type} : {e}")
        
        # Création et démarrage du thread
        thread = threading.Thread(
            target=process_messages, 
            name=f"{queue_type.capitalize()}MessageProcessor",
            daemon=True
        )
        thread.start()
        
        # Stocker la référence du thread
        if queue_type == 'clarification':
            self._clarification_thread = thread
        else:
            self._progress_thread = thread
        
        return thread
    
    def enqueue_clarification_request(
        self, 
        request: Dict[str, Any], 
        queue_type: str = 'clarification'
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
                self._clarification_queue if queue_type == 'clarification'
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
    
    def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """
        Récupère le statut d'une conversation.
        
        Args:
            conversation_id (str): ID de la conversation
        
        Returns:
            dict: Statut de la conversation
        """
        conversation = self._active_conversations.get(conversation_id)
        
        if not conversation:
            return {
                'status': 'not_found',
                'message': 'Conversation non trouvée'
            }
        
        return {
            'status': 'found',
            'request': conversation.get('request'),
            'response': conversation.get('response')
        }
    
    def start_processing(self):
        """
        Démarre le traitement des files de clarification et de progression.
        """
        # Démarrer les threads de traitement
        self.handle_clarification_request(queue_type='clarification')
        self.handle_clarification_request(queue_type='progress')
    
    def stop_processing(self):
        """
        Arrête proprement le traitement des demandes.
        """
        # Signaler l'arrêt
        self._stop_event.set()
        
        # Attendre la fin des threads
        if self._clarification_thread:
            self._clarification_thread.join(timeout=5)
        if self._progress_thread:
            self._progress_thread.join(timeout=5)
        
        logger.info("Arrêt du traitement des demandes")
    
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
    
    async def _process_progress_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un message de progression de tâche.
        
        Args:
            message (dict): Message de progression
        
        Returns:
            dict: Résultat du traitement de progression
        """
        try:
            # Extraire les informations de progression
            task_id = message.get('task_id')
            current_step = message.get('current_step')
            total_steps = message.get('total_steps')
            source_agent = message.get('source_agent', 'unknown')
            additional_info = message.get('additional_info', {})
            
            # Mettre à jour l'état des tâches en cours
            if task_id:
                self._ongoing_tasks[task_id] = {
                    'current_step': current_step,
                    'total_steps': total_steps,
                    'source_agent': source_agent,
                    'additional_info': additional_info
                }
            
            # Générer un message de progression utilisateur
            progress_message = (
                f"Progression de la tâche {task_id} par {source_agent} : "
                f"Étape {current_step}/{total_steps}"
            )
            
            # Ajouter des informations supplémentaires si disponibles
            if additional_info:
                progress_message += f"\nDétails : {additional_info}"
            
            return {
                'status': 'progress_processed',
                'task_id': task_id,
                'progress_message': progress_message
            }
        
        except Exception as e:
            logger.error(f"Erreur de traitement de la progression : {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
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
            connection = self._get_rabbitmq_connection()
            channel = connection.channel()
            
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

    def track_complex_task(
        self, 
        task_description: str, 
        subtasks: List[Dict[str, Any]]
    ) -> str:
        """
        Suit une tâche complexe décomposée en sous-tâches.
        
        Args:
            task_description (str): Description de la tâche principale
            subtasks (list): Liste des sous-tâches à suivre
        
        Returns:
            str: ID unique de la tâche complexe
        """
        # Générer un ID unique pour la tâche
        task_id = str(uuid.uuid4())
        
        # Préparer la structure de suivi de la tâche
        complex_task = {
            'task_id': task_id,
            'description': task_description,
            'total_subtasks': len(subtasks),
            'completed_subtasks': 0,
            'subtasks': subtasks,
            'status': 'in_progress',
            'created_at': datetime.now().isoformat()
        }
        
        # Stocker la tâche complexe
        self._ongoing_tasks[task_id] = complex_task
        
        # Publier un message de progression initial
        initial_progress_message = {
            'task_id': task_id,
            'current_step': 0,
            'total_steps': len(subtasks),
            'status': 'started',
            'description': task_description
        }
        
        # Ajouter à la file de progression
        self.enqueue_clarification_request(
            initial_progress_message, 
            queue_type='progress'
        )
        
        return task_id
    
    def update_complex_task(
        self, 
        task_id: str, 
        subtask_index: int, 
        status: str = 'completed',
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Met à jour l'état d'une sous-tâche dans une tâche complexe.
        
        Args:
            task_id (str): ID de la tâche complexe
            subtask_index (int): Index de la sous-tâche
            status (str): Statut de la sous-tâche
            additional_info (dict, optional): Informations supplémentaires
        
        Returns:
            dict: État mis à jour de la tâche
        """
        # Vérifier si la tâche existe
        if task_id not in self._ongoing_tasks:
            raise ValueError(f"Tâche {task_id} non trouvée")
        
        # Récupérer la tâche
        task = self._ongoing_tasks[task_id]
        
        # Mettre à jour la sous-tâche
        if 0 <= subtask_index < len(task['subtasks']):
            task['subtasks'][subtask_index]['status'] = status
            
            # Mettre à jour le nombre de sous-tâches complétées
            task['completed_subtasks'] = sum(
                1 for st in task['subtasks'] if st.get('status') == 'completed'
            )
            
            # Vérifier si toutes les sous-tâches sont terminées
            if task['completed_subtasks'] == task['total_subtasks']:
                task['status'] = 'completed'
            
            # Préparer le message de progression
            progress_message = {
                'task_id': task_id,
                'current_step': task['completed_subtasks'],
                'total_steps': task['total_subtasks'],
                'status': task['status'],
                'description': task['description'],
                'additional_info': additional_info or {}
            }
            
            # Ajouter à la file de progression
            self.enqueue_clarification_request(
                progress_message, 
                queue_type='progress'
            )
            
            return task
        
        raise ValueError(f"Sous-tâche {subtask_index} invalide pour la tâche {task_id}")
    
    def get_complex_task_status(
        self, 
        task_id: str
    ) -> Dict[str, Any]:
        """
        Récupère le statut d'une tâche complexe.
        
        Args:
            task_id (str): ID de la tâche complexe
        
        Returns:
            dict: Statut détaillé de la tâche
        """
        task = self._ongoing_tasks.get(task_id)
        
        if not task:
            return {
                'status': 'not_found',
                'message': f'Tâche {task_id} non trouvée'
            }
        
        return {
            'status': 'found',
            'task_id': task_id,
            'description': task['description'],
            'total_subtasks': task['total_subtasks'],
            'completed_subtasks': task['completed_subtasks'],
            'progress_percentage': (task['completed_subtasks'] / task['total_subtasks']) * 100,
            'task_status': task['status'],
            'subtasks': task['subtasks']
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
    agent.start_processing()
    
    return agent
