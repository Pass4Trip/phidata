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

from phi.agent import Agent, AgentMemory
from phi.model.openai import OpenAIChat
from phi.storage.agent.postgres import PgAgentStorage
from phi.memory.db.postgres import PgMemoryDb

from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from sqlalchemy import text

import threading
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
    import pika
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
    
    # Nom de la queue de retour
    result_queue_name = 'queue_retour_orchestrator'
    
    try:
        # Établir la connexion
        with pika.BlockingConnection(connection_params) as connection:
            channel = connection.channel()
            
            # Déclarer la queue si elle n'existe pas
            channel.queue_declare(queue=result_queue_name, durable=True)
            
            # Temps de début
            start_time = time.time()
            
            while (time.time() - start_time) < timeout:
                # Récupérer un message
                method_frame, properties, body = channel.basic_get(queue=result_queue_name)
                
                if method_frame is None:
                    # Pas de message, attendre un peu
                    time.sleep(poll_interval)
                    continue
                
                try:
                    # Décoder le message
                    message = json.loads(body.decode('utf-8'))
                    
                    # Vérifier si c'est le bon message
                    if (message.get('session_id') == session_id and 
                        message.get('status') == 'completed'):
                        
                        # Acquitter le message
                        channel.basic_ack(method_frame.delivery_tag)
                        
                        # Log du message complet
                        logger.info(f"Message de complétion reçu pour la session {session_id} :")
                        logger.info(json.dumps(message, indent=2))
                        
                        # Retourner le message complet
                        return json.dumps(message, indent=2)
                    
                    # Si pas le bon message, le remettre dans la queue
                    channel.basic_nack(method_frame.delivery_tag, requeue=True)
                
                except json.JSONDecodeError:
                    logger.error(f"Erreur de décodage JSON pour le message : {body}")
                    channel.basic_nack(method_frame.delivery_tag, requeue=False)
                
                # Attendre un peu avant la prochaine vérification
                time.sleep(poll_interval)
            
            # Timeout atteint
            return f"Timeout : pas de réponse pour la session {session_id} dans le délai imparti"
    
    except Exception as e:
        logger.error(f"Erreur lors de l'attente de la complétion de tâche : {e}")
        return f"Erreur : {str(e)}"


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
        message_retour = {
            "session_id": current_session_id,
            "status": "completed", 
            "query": query, 
            "result": "Résultat détaillé de la tâche",
            "metadata": { 
                "sources": ["RabbitMQ"], 
                "timestamp": datetime.now().isoformat() 
            }
        }
        
        # Log du message de retour attendu
        logger.info(f"Message de retour attendu : {json.dumps(message_retour, indent=2)}")
        
        # Attendre la complétion de la tâche
        return wait_for_task_completion(session_id=current_session_id)
        
    # Configuration de l'agent
    agent_base = Agent(
        instructions=[
            "Tu es un agent spécialisé dans la transmission de messages via RabbitMQ.",
            "Workflow de traitement des requêtes :",
            "1. Réception d'une requête à transmettre",
            "2. Génération d'un identifiant de session unique",
            "3. Envoi du message à RabbitMQ avec les informations suivantes :",
            "   - session_id : identifiant unique de session",
            "   - query : contenu original de la requête",
            "   - user_id : identifiant de l'utilisateur (si disponible)",
            "   - timestamp : horodatage de l'envoi",
            "4. Sauvegarde de la tâche en base de données",
            "5. Attente du message de complétion dans la queue 'queue_retour_orchestrator'",
            "",
            "CONSIGNES IMPORTANTES :",
            "- NE JAMAIS traiter ou répondre à la requête originale",
            "- Utiliser UNIQUEMENT la fonction submit_task(query)",
            "- Exemple : submit_task(query='Quel est le fonctionnement du moteur électrique ?')",
            "- Le résultat sera récupéré automatiquement via la queue de retour"
        ],
        model=OpenAIChat(
            model=model_id,
            temperature=0.3,  # Température basse pour des réponses plus déterministes
            max_tokens=150  # Limiter la longueur des réponses
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
