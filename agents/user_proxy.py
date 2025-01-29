from typing import Optional, Any, Dict, Callable
import os
import logging
from dotenv import load_dotenv
import json
import uuid
from datetime import datetime, timedelta
from datetime import datetime
from typing import Optional
import pika
import threading

from phi.agent import Agent, AgentMemory
from phi.model.openai import OpenAIChat
from phi.storage.agent.postgres import PgAgentStorage
from phi.memory.db.postgres import PgMemoryDb

from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from sqlalchemy import text

import queue
from typing import Callable, Any, Optional
from datetime import datetime, timedelta

Base = declarative_base()

class UserProxyTask(Base):
    """
    Modèle SQLAlchemy pour la table userproxy_task_memories
    """
    __tablename__ = 'userproxy_task_memories'

    task_id = Column(String, primary_key=True, nullable=False)
    user_id = Column(String, nullable=False)
    task_state = Column(String, nullable=False)
    task_desc = Column(Text, nullable=True)
    task_create_on = Column(DateTime, default=datetime.now)
    task_closed_on = Column(DateTime, nullable=True)

def save_user_proxy_task(
    task_id: str, 
    user_id: str, 
    task_state: str, 
    task_desc: Optional[str] = None,
    task_create_on: Optional[datetime] = None,
    task_closed_on: Optional[datetime] = None
) -> bool:
    """
    Enregistre une tâche dans la table userproxy_task_memories.
    
    Args:
        task_id (str): Identifiant unique de la tâche
        user_id (str): Identifiant de l'utilisateur
        task_state (str): État de la tâche
        task_desc (Optional[str]): Description de la tâche
        task_create_on (Optional[datetime]): Date de création de la tâche
        task_closed_on (Optional[datetime]): Date de fermeture de la tâche
    
    Returns:
        bool: True si l'enregistrement a réussi, False sinon
    """
    import logging
    from sqlalchemy.exc import SQLAlchemyError, IntegrityError
    
    logger = logging.getLogger('agents.user_proxy')
    
    # Validation des paramètres d'entrée
    if not task_id or not user_id or not task_state:
        logger.error("Paramètres invalides : task_id, user_id et task_state sont requis")
        return False
    
    try:
        # Créer le moteur de base de données
        engine = create_engine(db_url, echo=False)
        
        # Créer les tables si elles n'existent pas
        Base.metadata.create_all(engine)
        
        # Créer une session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Créer l'objet tâche
        new_task = UserProxyTask(
            task_id=task_id,
            user_id=user_id,
            task_state=task_state,
            task_desc=task_desc,
            task_create_on=task_create_on or datetime.now(),
            task_closed_on=task_closed_on
        )
        
        # Ajouter et commiter la transaction
        session.add(new_task)
        session.commit()
        
        logger.info(f"Tâche {task_id} enregistrée avec succès pour l'utilisateur {user_id}")
        return True
    
    except IntegrityError as ie:
        logger.error(f"Erreur d'intégrité lors de l'enregistrement de la tâche : {str(ie)}")
        session.rollback()
        return False
    
    except SQLAlchemyError as se:
        logger.error(f"Erreur SQLAlchemy lors de l'enregistrement de la tâche : {str(se)}")
        session.rollback()
        return False
    
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'enregistrement de la tâche : {str(e)}")
        return False
    
    finally:
        # Fermer la session
        if 'session' in locals():
            session.close()


# Charger les variables d'environnement
load_dotenv()

# Construction dynamique de l'URL de base de données PostgreSQL
def build_postgres_url() -> str:
    """
    Construire dynamiquement l'URL de connexion PostgreSQL à partir des variables d'environnement
    
    Returns:
        str: URL de connexion PostgreSQL
    """
    # Récupérer les paramètres de connexion depuis les variables d'environnement
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


def get_user_preferences(
    query: str,
    user_id: str, 
    db_url: Optional[str] = None, 
    table_name: str = "user_proxy_memories"
) -> Dict[str, Any]:
    """
    Récupère les mémoires d'un utilisateur à partir de la mémoire Phidata.
    
    Args:
        user_id (str): Identifiant de l'utilisateur
        db_url (Optional[str]): URL de connexion à la base de données PostgreSQL
        table_name (str): Nom de la table de mémoire
    
    Returns:
        List[str]: Liste des mémoires de l'utilisateur
    """
    # Utiliser l'URL de base de données globale si non fournie
    if db_url is None:
        db_url = globals().get('db_url')
    
    if not db_url:
        logger.error("Aucune URL de base de données fournie.")
        return []
    
    try:
        # Créer une mémoire d'agent avec la base de données PostgreSQL
        agent_memory = AgentMemory(
            db=PgMemoryDb(
                table_name=table_name, 
                db_url=db_url
            ),
            # Définir l'ID utilisateur
            user_id=user_id
        )
        
        # Charger les mémoires de l'utilisateur
        memories = agent_memory.db.read_memories(user_id=user_id)
        
        # Stocker les mémoires utilisateur
        user_memories = []
        
        # Parcourir toutes les mémoires
        for memory in memories:
            try:
                memory_content = None
                
                if isinstance(memory.memory, dict):
                    memory_content = memory.memory.get('memory')
                elif isinstance(memory.memory, str):
                    try:
                        memory_dict = json.loads(memory.memory)
                        memory_content = memory_dict.get('memory')
                    except json.JSONDecodeError:
                        memory_content = memory.memory
                else:
                    memory_content = str(memory.memory)
                
                if memory_content:
                    user_memories.append(memory_content)
            
            except Exception as e:
                logger.error(f"❌ Erreur lors du traitement de la mémoire : {e}")
                pass
        
        logger.info(f"📋 Nombre de mémoires extraites : {len(user_memories)}")
        
        return user_memories
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des mémoires : {e}")
        return []


class UserTaskManager:
    """
    Gestionnaire avancé de tâches utilisateur avec intégration Phidata
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.task_queues = {}
            self.active_tasks = {}
            self.logger = logging.getLogger('agents.task_manager')
            self.initialized = True

    def enqueue_task(
        self, 
        user_id: str, 
        task_function: Callable[[], Any],
        max_queue_size: int = 5,
        task_timeout: int = 600  # 10 minutes
    ) -> dict:
        """
        Ajouter une tâche à la file d'un utilisateur
        """
        with self._lock:
            # Initialiser la file si nécessaire
            if user_id not in self.task_queues:
                self.task_queues[user_id] = queue.Queue(maxsize=max_queue_size)
            
            # Vérifier la taille de la file
            if self.task_queues[user_id].full():
                self.logger.warning(f"File d'attente pleine pour {user_id}")
                return {
                    'status': 'error',
                    'message': 'Trop de tâches en attente',
                    'code': 'QUEUE_FULL'
                }
            
            # Générer un identifiant de tâche
            task_id = str(uuid.uuid4())
            
            # Préparer les informations de la tâche
            task_info = {
                'task_id': task_id,
                'user_id': user_id,
                'status': 'queued',
                'created_at': datetime.now(),
                'timeout_at': datetime.now() + timedelta(seconds=task_timeout)
            }
            
            # Ajouter à la file
            self.task_queues[user_id].put((task_info, task_function))
            
            # Démarrer le traitement si pas de tâche active
            if user_id not in self.active_tasks:
                self._start_task_processing(user_id)
            
            return {
                'status': 'success',
                'task_id': task_id,
                'message': 'Tâche en file d\'attente'
            }

    def _start_task_processing(self, user_id: str):
        """
        Démarrer le traitement des tâches pour un utilisateur
        """
        def task_processor():
            while True:
                try:
                    # Récupérer la prochaine tâche
                    with self._lock:
                        if self.task_queues[user_id].empty():
                            del self.active_tasks[user_id]
                            break
                        
                        task_info, task_function = self.task_queues[user_id].get()
                    
                    # Vérifier le timeout
                    if datetime.now() > task_info['timeout_at']:
                        self.logger.warning(f"Tâche {task_info['task_id']} expirée")
                        task_info['status'] = 'timeout'
                        continue
                    
                    # Exécuter la tâche
                    try:
                        self.logger.info(f"Début de la tâche {task_info['task_id']}")
                        task_info['status'] = 'processing'
                        result = task_function()
                        task_info['status'] = 'completed'
                        task_info['result'] = result
                    except Exception as e:
                        self.logger.error(f"Erreur lors de l'exécution de la tâche : {e}")
                        task_info['status'] = 'failed'
                        task_info['error'] = str(e)
                    
                    # Marquer la tâche comme traitée
                    self.task_queues[user_id].task_done()
                
                except Exception as e:
                    self.logger.critical(f"Erreur critique dans le processeur de tâches : {e}")
                    break
        
        # Démarrer le thread de traitement
        thread = threading.Thread(target=task_processor, daemon=True)
        thread.start()
        self.active_tasks[user_id] = thread

    def get_task_status(self, user_id: str, task_id: str) -> Optional[dict]:
        """
        Récupérer le statut d'une tâche
        """
        if user_id not in self.task_queues:
            return None
        
        for queue_item in list(self.task_queues[user_id].queue):
            if queue_item[0]['task_id'] == task_id:
                return queue_item[0]
        return None

# Singleton global pour le gestionnaire de tâches
global_task_manager = UserTaskManager()


def send2RabbitMQ(
    query: str = "query text", 
    session_id: Optional[str] = None, 
    user_id: Optional[str] = None,
    date: Optional[str] = None
) -> str:
    """
    Envoie un message à RabbitMQ avec gestion des erreurs et logging.
    
    :param query: Le message à envoyer
    :param session_id: L'identifiant de session
    :param user_id: L'identifiant de l'utilisateur
    :return: Un message décrivant le résultat de l'envoi
    """
    import pika
    import os
    import logging
    import json
    import uuid
    from datetime import datetime
    
    try:
        # Configuration par défaut RabbitMQ
        config = {
            'host': os.getenv('RABBITMQ_HOST', 'localhost'),
            'port': int(os.getenv('RABBITMQ_PORT', 5672)),
            'credentials': pika.PlainCredentials(
                os.getenv('RABBITMQ_USER', 'guest'),
                os.getenv('RABBITMQ_PASSWORD', 'guest')
            )
        }
        queue_name = 'test_user_proxy'
        
        # Nettoyer et sécuriser les paramètres
        query = query.strip() if query else "Message vide reçu"
   
        
        # Préparer le message
        message = {
            'session_id': session_id,
            'user_id': user_id,
            'query': query,
            'status': 'in_progress',
            'timestamp': date
        }
        
        # Établir la connexion et envoyer
        with pika.BlockingConnection(pika.ConnectionParameters(**config)) as connection:
            channel = connection.channel()
            channel.queue_declare(queue=queue_name, durable=True)
            
            channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(message).encode('utf-8'),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Message persistant
                    content_type='application/json'
                )
            )
            
            logger.info(f"Message envoyé à RabbitMQ sur la file {queue_name}")
            return f"Message envoyé avec succès sur la file {queue_name}"
    
    except Exception as e:
        error_msg = f"Erreur lors de l'envoi du message à RabbitMQ : {str(e)}"
        logger.error(error_msg)
        return error_msg


def wait_for_task_completion(
    session_id: str, 
    timeout: int = 600,  # 10 minutes par défaut
    poll_interval: float = 0.5  # Intervalle de vérification en secondes
) -> str:
    """
    Attendre la complétion d'une tâche via RabbitMQ.
    
    Args:
        session_id (str): Identifiant de session unique
        timeout (int): Temps maximum d'attente en secondes
        poll_interval (float): Intervalle entre chaque vérification
    
    Returns:
        str: Résultat de la tâche ou message d'erreur
    """

    import json
    import time
    import logging

    
    logger = logging.getLogger('agents.user_proxy')
    
    # Paramètres de connexion RabbitMQ
    connection_params = pika.ConnectionParameters(
        host=os.getenv('RABBITMQ_HOST', 'localhost'),
        port=int(os.getenv('RABBITMQ_PORT', 5672)),
        credentials=pika.PlainCredentials(
            os.getenv('RABBITMQ_USER', 'guest'),
            os.getenv('RABBITMQ_PASSWORD', 'guest')
        )
    )
    
    # Noms des queues
    result_queue_name = 'queue_retour_orchestrator'
    chatbot_queue_name = 'queue_chatbot'
    
    try:
        logger.info(f" Démarrage de wait_for_task_completion pour session {session_id}")
        logger.info(f" Paramètres de connexion RabbitMQ : {connection_params}")
        
        # Établir la connexion
        with pika.BlockingConnection(connection_params) as connection:
            channel = connection.channel()
            
            # Déclarer les queues si elles n'existent pas
            logger.info(f" Déclaration des queues : {result_queue_name}, {chatbot_queue_name}")
            channel.queue_declare(queue=result_queue_name, durable=True)
            channel.queue_declare(queue=chatbot_queue_name, durable=True)
            
            # Temps de début
            start_time = time.time()
            
            logger.info(f" Début de la boucle d'attente. Timeout : {timeout} secondes")
            while (time.time() - start_time) < timeout:
                # Récupérer un message
                method_frame, properties, body = channel.basic_get(queue=result_queue_name)
                
                if method_frame is None:
                    # Pas de message, attendre un peu
                    logger.debug(f" Aucun message dans la queue {result_queue_name}. Attente...")
                    time.sleep(poll_interval)
                    continue
                
                try:
                    # Décoder le message en supprimant les caractères de retour chariot et les espaces supplémentaires
                    message_str = body.decode('utf-8').strip()
                    
                    logger.info(f" Message brut reçu : {message_str[:200]}...")
                    
                    # Nettoyer manuellement la chaîne JSON
                    message_str = message_str.replace('\r\n', '').replace('\n', '')
                    
                    # Ajouter des virgules manquantes si nécessaire
                    if message_str.startswith(' {'):
                        message_str = '{' + message_str[2:]
                    
                    # Tenter de décoder le JSON
                    message = json.loads(message_str)
                    
                    logger.info(f" Message JSON décodé : {json.dumps(message, indent=2)}")
                    
                    # Vérifier si c'est le bon message
                    if (message.get('session_id') == session_id and 
                        message.get('status') == 'completed'):
                        
                        logger.info(f" Message correspondant trouvé pour la session {session_id}")
                        
                        # Acquitter le message de la queue originale
                        channel.basic_ack(method_frame.delivery_tag)
                        
                        # Transférer le message dans queue_chatbot
                        logger.info(f" Transfert du message vers {chatbot_queue_name}")
                        channel.basic_publish(
                            exchange='',
                            routing_key=chatbot_queue_name,
                            body=json.dumps(message),
                            properties=pika.BasicProperties(
                                delivery_mode=2  # Message persistant
                            )
                        )
                        
                        # Log du message complet
                        logger.info(f" Message de complétion final pour la session {session_id}")
                        logger.info(json.dumps(message, indent=2))
                        
                        # Retourner le message complet
                        return json.dumps(message, indent=2)
                    
                    # Si pas le bon message, le remettre dans la queue
                    logger.info(f" Message ne correspondant pas à la session {session_id}. Remis en queue.")
                    channel.basic_nack(method_frame.delivery_tag, requeue=True)
                
                except json.JSONDecodeError as e:
                    # Log de l'erreur détaillée
                    logger.error(f" Erreur de décodage JSON : {e}")
                    logger.error(f" Message brut reçu : {body}")
                    logger.error(f" Message nettoyé : {message_str}")
                    
                    # Tenter un décodage plus permissif
                    try:
                        # Utiliser un décodeur JSON plus permissif
                        import ast
                        message = ast.literal_eval(body.decode('utf-8'))
                        
                        # Si le décodage réussit, convertir en JSON
                        message_json = json.dumps(message)
                        logger.info(f" Décodage réussi avec ast.literal_eval : {message_json}")
                    except Exception as ast_error:
                        logger.error(f" Échec du décodage avec ast.literal_eval : {ast_error}")
                    
                    channel.basic_nack(method_frame.delivery_tag, requeue=False)
                
                # Attendre un peu avant la prochaine vérification
                time.sleep(poll_interval)
            
            # Timeout atteint
            logger.warning(f" Timeout atteint pour la session {session_id}")
            return f"Timeout : pas de réponse pour la session {session_id} dans le délai imparti"
    
    except Exception as e:
        logger.error(f" Erreur lors de l'attente de la complétion de tâche : {e}")
        return f"Erreur : {str(e)}"
    
    finally:
        logger.info(f" Fin de wait_for_task_completion pour la session {session_id}")


def enrich_query(self, query: str) -> str:
    """
    Enrichit la requête utilisateur avec des informations pertinentes de la mémoire

    Args:
        query (str): La requête utilisateur originale

    Returns:
        str: La requête enrichie
    """
    logger.info(f"🔍 Enrichissement de la requête : '{query}'")
    
    try:
        # Récupérer les informations pertinentes de la mémoire
        relevant_info = self.memory.get_relevant(query)
        
        # Préparer le contexte pour l'enrichissement
        context = "\n".join(relevant_info)
        
        # Utiliser l'API OpenAI pour enrichir la requête
        response = self._openai_client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system", 
                    "content": """
                    Tu es un expert en enrichissement de requêtes.
                    Utilise le contexte fourni pour enrichir la requête de manière pertinente.
                    Garde l'essentiel de la requête originale, mais ajoute des détails ou précisions utiles.
                    """
                },
                {
                    "role": "user", 
                    "content": f"Requête originale : {query}\n\nContexte :\n{context}"
                }
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        enriched_query = response.choices[0].message.content.strip()
        logger.info(f"Requête enrichie : '{enriched_query}'")
        
        return enriched_query
    
    except Exception as e:
        logger.error(f"Erreur lors de l'enrichissement de la requête : {e}")
        return query  # Retourner la requête originale en cas d'erreur



def get_user_proxy_agent(
    model_id: str = "gpt-4o-mini",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
    **kwargs
) -> Agent:
    """
    Créer et configurer l'agent User Proxy.
    
    Args:
        model_id (str): Identifiant du modèle OpenAI à utiliser
        user_id (Optional[str]): Identifiant de l'utilisateur
        session_id (Optional[str]): Identifiant de session
        debug_mode (bool): Mode débogage
    
    Returns:
        Agent: Agent User Proxy configuré
    """
    def submit_task(query: str) -> str:
        """
        Soumettre une tâche pour l'utilisateur courant
        
        Args:
            query (str): Requête à traiter
        
        Returns:
            str: Résultat de la soumission de tâche
        """
        # Générer un nouvel identifiant de session
        current_session_id = str(uuid.uuid4())
        
        # Date courante
        date = datetime.now().isoformat()
        
        # Créer un événement pour signaler la complétion
        task_completed = threading.Event()
        task_result = [None]
        
        def wait_for_completion():
            """
            Fonction pour attendre la complétion en arrière-plan
            """
            try:
                result = wait_for_task_completion(session_id=current_session_id)
                task_result[0] = result
            except Exception as e:
                task_result[0] = f"Erreur : {str(e)}"
            finally:
                task_completed.set()
        
        # Log avant le démarrage du thread
        logger.info(f" Préparation du thread de completion pour la tâche {current_session_id}")
        
        # Démarrer le thread d'attente
        completion_thread = threading.Thread(
            target=wait_for_completion, 
            daemon=True  # Thread en arrière-plan
        )
        completion_thread.start()
        
        # Log après le démarrage du thread
        logger.info(f" Thread de completion démarré pour la tâche {current_session_id}")
        
        # Vérification supplémentaire de l'état du thread
        if completion_thread.is_alive():
            logger.info(f" Thread de completion actif pour la tâche {current_session_id}")
        else:
            logger.warning(f" Le thread de completion ne semble pas actif pour la tâche {current_session_id}")
        
        query = get_user_preferences(query, user_id=user_id, db_url=db_url)

        # Envoi à RabbitMQ
        send2RabbitMQ(
            query=query, 
            session_id=current_session_id, 
            user_id=user_id, 
            date=date
        )
        
        # Sauvegarde de la tâche
        save_user_proxy_task(
            task_id=current_session_id, 
            user_id=user_id or 'anonymous', 
            task_state='in_progress', 
            task_desc=query,
            task_create_on=datetime.fromisoformat(date)
        )
        
        # Log d'attente du message de complétion
        logger.info(f"En attente du message de complétion pour la session {current_session_id}")
        
        # Préparer le message de retour attendu
        # message_retour = {
        #     "session_id": current_session_id,
        #     "status": "completed", 
        #     "query": query, 
        #     "result": "Résultat détaillé de la tâche",
        #     "metadata": { 
        #         "sources": ["RabbitMQ"], 
        #         "timestamp": datetime.now().isoformat() 
        #     }
        # }
        
        # # Log du message de retour attendu
        # logger.info(f"Message de retour attendu : {json.dumps(message_retour, indent=2)}")
        
        # Retourner immédiatement un message indiquant que la tâche est en cours
        return json.dumps(f"Tâche {current_session_id} en cours de traitement")
        
    
        # Configuration de l'agent
    agent_base = Agent(
        instructions=[
            "Tu es un agent intelligent avec deux modes de fonctionnement : transmission et traitement direct.",
            "",
            "- Règles de routage entre les 2 modes :",
            "  * Mode TRAITEMENT DIRECT :",
            "    - Pour les questions simples, générales ou relatives aux infos de l'utilisateur",
            "    - Exemples : salutations, préférences, informations basiques",
            "  * Mode TRANSMISSION DE REQUÊTE :",
            "    - Pour les demandes complexes nécessitant :",
            "      1. Recherche web",
            "      2. Décomposition de la tâche",
            "      3. Utilisation d'outils spécifiques",
            "    - Exemples : analyse de données, recherches approfondies, tâches multi-étapes",
            "MODE 1 : TRANSMISSION DE REQUÊTE",
            "- Objectif : Transmettre des requêtes complexes à un système de traitement",
            "- Workflow :",
            "  1. Réceptionner une requête nécessitant un traitement approfondi",
            "  2. Générer un identifiant de session unique",
            "  3. Préparer les métadonnées :",
            "     * session_id : identifiant unique",
            "     * query : contenu original",
            "     * user_id : identifiant utilisateur",
            "     * timestamp : horodatage",
            "  4. Envoyer le message via RabbitMQ",
            "  5. Sauvegarder la tâche en base de données",
            "  6. Attendre le message de complétion dans 'queue_retour_orchestrator'",
            "- Exemples de requêtes en mode transmission :",
            "  * 'Quel est le fonctionnement du moteur électrique ?'",
            "  * 'Résume le dernier rapport technique'",
            "- En mode transmission, utiliser submit_task(query)",
            "", 
            "",
            "MODE 2 : TRAITEMENT DIRECT",
            "- Objectif : Répondre rapidement aux requêtes simples et gérer les interactions directes",
            "- Workflow pour le traitement direct :",
            "  1. Analyse la requête de l'utilisateur",
            "  2. Détermine si une réponse directe est appropriée",
            "  3. Formule une réponse adaptée en utilisant le style de communication défini",
            "  4. Envoie la réponse à l'utilisateur",
            "- Style de communication :",
            "  * Sois sympathique et naturel dans tes réponses",
            "  * Garde un ton léger mais professionnel",
            "  * Utilise quelques emojis pour illustrer tes messages",
            "  * Important : En tant que LLM de l'agent User Proxy, tu es responsable de générer",
            "  * la réponse finale à l'utilisateur. Assure-toi que ta réponse soit complète,",
            "  * pertinente et respecte le style de communication demandé.",
            "  * Sois sympathique et naturel dans tes réponses",
            "  * Garde un ton léger mais professionnel",
            "  * Utilise quelques emojis pour illustrer tes messages",
            "- Exemples de requêtes en mode direct :",
            "  * 'Quelles est la date de naissance de Henri 4 roi de France ?'",
            "  * 'Quelle est la capitale de la France ?'",
            "- En mode traitement direct, utilise au maximulm tes connaisances sur l'utuilisateur pour répondre",
            "- Capacité à comprendre et identifier des dates relatives :",
            "  * 'hier', 'aujourd'hui', 'demain'",
            "  * 'la semaine dernière', 'le mois prochain', 'l'année prochaine'",
            "  * 'dans 3 jours', 'il y a 2 semaines'",
            "  * 'le premier lundi du mois prochain'",
            "- Exemples de requêtes de date :",
            "  * 'Quelle est la date d'aujourd'hui ?'",
            "  * 'Donne-moi la date du dimanche dans 2 semaines'",
            "  * 'Quel jour serons-nous dans 10 jours ?'",
            "  * 'Quelle était la date il y a 3 mois ?'",
            "- Exemple de calcul de date :",
            "  Si aujourd'hui nous sommes le mercredi 29 janvier 2025,",
            "  alors 'le deuxième mardi du mois prochain' sera le mardi 11 février 2025",
        ],
        model=OpenAIChat(
            model=model_id,
            temperature=0.3,  # Température basse pour des réponses plus déterministes
            max_tokens=500  # Limiter la longueur des réponses
        ),
        tools=[
            submit_task,  # Nouvelle méthode de soumission de tâche
            # Conserver les autres outils existants si nécessaire
        ],
        debug_mode=debug_mode,
        agent_id="user_proxy_agent",
        user_id=user_id,
        session_id=session_id,
        name="User Proxy Agent",
        add_history_to_messages=True,
        num_history_responses=10,
        memory=AgentMemory(
            db=PgMemoryDb(table_name="user_proxy_memories", db_url=db_url),
            create_user_memories=True,
            update_user_memories_after_run=True,
            create_session_summary=True,
            update_session_summary_after_run=True,
        ),        
        storage=PgAgentStorage(table_name="user_proxy_sessions", db_url=db_url),
        **kwargs
    )
    
    return agent_base
