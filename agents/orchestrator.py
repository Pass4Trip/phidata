import os
import sys
import asyncio
import logging
import json
import uuid
import traceback
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from phi.agent import RunResponse, Agent
import openai

from agents.web import get_web_searcher
from agents.settings import agent_settings
from agents.orchestrator_prompts import (
    get_task_decomposition_prompt,
    get_task_execution_prompt,
    get_task_synthesis_prompt
)

from utils.colored_logging import get_colored_logger

# Ajouter le répertoire parent au PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ajout d'un handler de console si nécessaire
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Ajouter le handler au logger s'il n'est pas déjà présent
if not logger.handlers:
    logger.addHandler(console_handler)

AGENT_ROUTING_PROMPT = """
Pour la tâche : '{task}'
Et les agents suivants :
{agents_description}

Historique des performances :
{performance_history}

Quel agent est le plus approprié pour cette tâche ?
Fournir le nom de l'agent, le score de confiance et le raisonnement.
Format : JSON avec les clés 'selected_agent', 'confidence_score' et 'reasoning'
"""

TASK_CONTEXT_PROMPT = """
Pour la tâche : '{task}'
Et le contexte suivant :
{context}

Analyser le contexte et fournir les éléments suivants :
- Une analyse du contexte
- Une approche recommandée
- Des considérations critiques
Format : JSON avec les clés 'context_analysis', 'recommended_approach' et 'critical_considerations'
"""

@dataclass
class TaskLedger:
    """
    Registre pour gérer les faits et le plan de tâches
    """
    original_request: str
    initial_plan: List[str] = field(default_factory=list)
    _current_plan: List[str] = field(default_factory=list, repr=False)
    facts: Dict[str, Any] = field(default_factory=lambda: {
        "verified": [],  # Faits vérifiés
        "to_lookup": [], # Faits à rechercher
        "derived": [],   # Faits dérivés (calculs/logique)
        "guesses": []    # Suppositions éduquées
    })
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self._current_plan, list):
            if isinstance(self._current_plan, dict):
                self._current_plan = list(self._current_plan.values())
            elif isinstance(self._current_plan, str):
                self._current_plan = [self._current_plan]
            else:
                self._current_plan = [str(self._current_plan)] if self._current_plan is not None else []

    @property
    def current_plan(self) -> List[str]:
        return self._current_plan

    @current_plan.setter
    def current_plan(self, value: Union[List[str], Dict[str, Any], str, Any]):
        if not isinstance(value, list):
            if isinstance(value, dict):
                value = list(value.values())
            elif isinstance(value, str):
                value = [value]
            else:
                value = [str(value)] if value is not None else []
        self._current_plan = value
        self.updated_at = datetime.now()

    def add_fact(self, fact: str, fact_type: str = "verified"):
        """Ajouter un fait au registre"""
        if fact_type in self.facts:
            if fact not in self.facts[fact_type]:
                self.facts[fact_type].append(fact)
                self.updated_at = datetime.now()

    def add_task(self, task: str):
        """Ajouter une tâche au plan courant"""
        if task not in self._current_plan:
            self._current_plan.append(task)
            self.updated_at = datetime.now()

    def register_agent(self, agent: Agent, agent_name: str):
        """Enregistrer un agent dans le registre"""
        self.context[agent_name] = agent

    def to_json(self) -> Dict[str, Any]:
        """
        Convertir le TaskLedger en format JSON
        """
        return {
            "original_request": self.original_request,
            "initial_plan": self.initial_plan,
            "current_plan": self.current_plan,
            "facts": self.facts,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "context": self.context
        }

@dataclass
class ProgressLedger:
    """
    Registre pour suivre la progression et gérer les blocages
    """
    task_ledger: TaskLedger
    completed_tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    task_history: List[Dict[str, Any]] = field(default_factory=list)
    agent_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stall_count: int = 0
    max_stalls: int = 2
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"

    def is_task_complete(self) -> bool:
        """Vérifie si toutes les tâches sont terminées"""
        return len(self.task_ledger.current_plan) == 0

    def is_making_progress(self) -> bool:
        """Vérifie si des progrès sont réalisés"""
        if not self.task_history:
            return True
        last_update = datetime.fromisoformat(self.task_history[-1]["completed_at"])
        time_since_last_update = datetime.now() - last_update
        return time_since_last_update.seconds < 300  # 5 minutes

    def is_stalled(self) -> bool:
        """Vérifie si l'exécution est bloquée"""
        return self.stall_count >= self.max_stalls

    def complete_task(self, task: str, result: Dict[str, Any]):
        """Marquer une tâche comme terminée"""
        agent_name = result.get('agent', 'unknown')
        
        # Mettre à jour les performances de l'agent
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {
                'tasks_completed': 0,
                'success_rate': 1.0,
                'total_tasks': 0,
                'recent_performance': []
            }
        
        performance = self.agent_performance[agent_name]
        performance['total_tasks'] += 1
        
        if 'error' not in result:
            performance['tasks_completed'] += 1
            performance['success_rate'] = performance['tasks_completed'] / performance['total_tasks']
            performance['recent_performance'].append('success')
        else:
            performance['recent_performance'].append('failure')
        
        # Stocker le résultat
        self.completed_tasks[task] = result
        self.task_history.append({
            "task": task,
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        
        # Retirer la tâche du plan courant
        if task in self.task_ledger.current_plan:
            self.task_ledger.current_plan.remove(task)
        
        self.updated_at = datetime.now()

    def increment_stall(self):
        """Incrémenter le compteur de blocages"""
        self.stall_count += 1
        self.updated_at = datetime.now()

    def reset_stall(self):
        """Réinitialiser le compteur de blocages"""
        self.stall_count = 0
        self.updated_at = datetime.now()

    def get_next_agent(self, task: str) -> str:
        """
        Sélectionner le prochain agent basé sur les performances
        """
        best_agent = None
        best_score = -1

        for agent_name, perf in self.agent_performance.items():
            score = perf['success_rate'] * (perf['tasks_completed'] + 1)
            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent or "API Knowledge Agent"  # Agent par défaut

    def to_json(self) -> Dict[str, Any]:
        """Convertir en format JSON"""
        def convert_result(result):
            if hasattr(result, 'content'):
                return {
                    "content": result.content,
                    "content_type": str(result.content_type),
                    "event": str(result.event)
                }
            return result

        return {
            "task_ledger": self.task_ledger.to_json(),
            "completed_tasks": {
                task: {k: convert_result(v) for k, v in result.items()} 
                for task, result in self.completed_tasks.items()
            },
            "task_history": [
                {
                    "task": entry["task"],
                    "result": convert_result(entry["result"]),
                    "completed_at": entry["completed_at"]
                } 
                for entry in self.task_history
            ],
            "agent_performance": self.agent_performance,
            "stall_count": self.stall_count,
            "status": self.status
        }

class OrchestratorAgent:
    """
    Agent orchestrateur avancé avec décomposition de tâches
    """
    def __init__(
        self,
        model_id: str = "gpt-4o-mini", 
        debug_mode: bool = False,
        original_request: Optional[str] = None,
        api_key: Optional[str] = None,  
        enable_web_agent: bool = True,
        enable_api_knowledge_agent: bool = False,
        enable_data_analysis_agent: bool = False,
        enable_travel_planner: bool = False
    ):
        """
        Initialiser l'agent orchestrateur avec des agents spécialisés
        
        Args:
            model_id (str): Identifiant du modèle OpenAI
            debug_mode (bool): Mode de débogage
            original_request (Optional[str]): Requête originale pour le TaskLedger
            api_key (Optional[str]): Clé API OpenAI personnalisée
            enable_web_agent (bool): Activer l'agent de recherche web
            enable_api_knowledge_agent (bool): Activer l'agent de connaissances API
            enable_data_analysis_agent (bool): Activer l'agent d'analyse de données
            enable_travel_planner (bool): Activer l'agent de planification de voyage
        """
        # Configuration du modèle LLM
        self.llm_config = {
            "model": model_id,
            "temperature": 0.2
        }
        
        # Utiliser la clé API fournie ou celle de l'environnement
        openai_api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        # Initialisation explicite du client OpenAI
        try:
            self.client = openai.OpenAI(api_key=openai_api_key)
        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation du client OpenAI : {e}")
            self.client = None
        
        # Initialiser le modèle OpenAI
        self.model = self.client.chat.completions.create
        
        # Initialiser self.llm comme alias de self.model
        self.llm = self.model
        
        # Initialisation centralisée des agents
        self.agents = self._initialize_specialized_agents(
            enable_web_agent=enable_web_agent,
            enable_api_knowledge_agent=enable_api_knowledge_agent,
            enable_data_analysis_agent=enable_data_analysis_agent,
            enable_travel_planner=enable_travel_planner
        )
        
        # Initialisation du mode de débogage
        self.debug_mode = debug_mode
        
        # Création du TaskLedger initial
        self.task_ledger = self._create_task_ledger(original_request)
        
        # Créer l'agent orchestrateur avec configuration simplifiée
        self.agent = self._create_orchestrator_agent(debug_mode)

    def _initialize_specialized_agents(
        self, 
        enable_web_agent: bool = False,
        enable_api_knowledge_agent: bool = False,
        enable_data_analysis_agent: bool = False,
        enable_travel_planner: bool = False
    ) -> Dict[str, Agent]:
        """
        Initialiser tous les agents spécialisés de manière dynamique
        
        Args:
            enable_web_agent (bool): Activer l'agent de recherche web
            enable_api_knowledge_agent (bool): Activer l'agent de connaissances API
            enable_data_analysis_agent (bool): Activer l'agent d'analyse de données
            enable_travel_planner (bool): Activer l'agent de planification de voyage
        
        Returns:
            Dict[str, Agent]: Dictionnaire des agents disponibles
        """
        logger.info("🤖 Initialisation des agents spécialisés")
        logger.info(f"🌐 Web Agent: {enable_web_agent}")
        logger.info(f"📚 API Knowledge Agent: {enable_api_knowledge_agent}")
        logger.info(f"📊 Data Analysis Agent: {enable_data_analysis_agent}")
        logger.info(f"✈️ Travel Planner Agent: {enable_travel_planner}")
        
        agents = {}
        
        # Agent de recherche web
        if enable_web_agent:
            try:
                agents["web_search"] = get_web_searcher(
                    model_id=agent_settings.gpt_4,
                    debug_mode=False,
                    name="Web Search Agent"
                )
                logger.info("✅ Agent de recherche web initialisé")
            except Exception as e:
                logger.error(f"❌ Erreur d'initialisation de l'agent de recherche web : {e}")
        
        # Agent de connaissances API (à implémenter)
        if enable_api_knowledge_agent:
            try:
                from agents.api_knowledge import get_api_knowledge_agent
                agents["api_knowledge"] = get_api_knowledge_agent(
                    debug_mode=False,
                    user_id=None,
                    session_id=None
                )
                logger.info("✅ Agent de connaissances API initialisé")
            except ImportError:
                logger.warning("❌ Agent de connaissances API non disponible")
        
        # Agent d'analyse de données (à implémenter)
        if enable_data_analysis_agent:
            try:
                from agents.data_analysis import get_data_analysis_agent
                agents["data_analysis"] = get_data_analysis_agent(
                    debug_mode=False,
                    user_id=None,
                    session_id=None
                )
                logger.info("✅ Agent d'analyse de données initialisé")
            except ImportError:
                logger.warning("❌ Agent d'analyse de données non disponible")
        
        # Agent de planification de voyage (à implémenter)
        if enable_travel_planner:
            try:
                from agents.travel import get_travel_planner_agent
                agents["travel_planner"] = get_travel_planner_agent(
                    debug_mode=False,
                    user_id=None,
                    session_id=None
                )
                logger.info("✅ Agent de planification de voyage initialisé")
            except ImportError:
                logger.warning("❌ Agent de planification de voyage non disponible")
        
        # Ajout d'un agent mathématique par défaut
        agents["math_agent"] = Agent(
            instructions=[
                "Tu es un agent spécialisé dans les calculs mathématiques.",
                "Réponds uniquement aux questions mathématiques simples.",
                "Assure-toi de donner une réponse précise et concise."
            ],
            name="Math Agent"
        )
        logger.info("✅ Agent mathématique par défaut initialisé")
        
        # Log des agents disponibles
        logger.info(f"🤖 Agents initialisés : {list(agents.keys())}")
        
        return agents

    def _create_task_ledger(self, original_request: Optional[str] = None) -> TaskLedger:
        """
        Créer un TaskLedger avec gestion robuste
        
        Args:
            original_request (Optional[str]): Requête originale
        
        Returns:
            TaskLedger: Registre de tâches initialisé
        """
        return TaskLedger(
            original_request=original_request or "Requête non spécifiée",
            context={name: agent for name, agent in self.agents.items()}
        )

    def _create_orchestrator_agent(self, debug_mode: bool = False) -> Agent:
        """
        Créer l'agent orchestrateur avec configuration simplifiée
        
        Args:
            debug_mode (bool): Mode de débogage
        
        Returns:
            Agent: Agent orchestrateur configuré
        """
        return Agent(
            # Utiliser la méthode create du client OpenAI
            llm=self.client.chat.completions.create,
            instructions=[
                "Tu es un agent d'orchestration avancé capable de décomposer des tâches complexes.",
                "Étapes de travail :",
                "1. Analyser la requête originale",
                "2. Décomposer la requête en sous-tâches précises et réalisables",
                "3. Attribuer chaque sous-tâche à l'agent le plus approprié",
                "4. Suivre la progression de chaque sous-tâche",
                "5. Intégrer et synthétiser les résultats partiels",
                "6. Adapter dynamiquement le plan si nécessaire"
            ],
            # Ajouter le mode de débogage si nécessaire
            debug_mode=debug_mode
        )

    def _get_task_decomposition_functions(self) -> List[Dict[str, Any]]:
        """
        Définir les fonctions d'appel pour la décomposition de tâches
        
        Returns:
            List[Dict[str, Any]]: Liste des définitions de fonctions
        """
        return [
            {
                "name": "decompose_task",
                "description": "Décomposer une tâche complexe en sous-tâches précises et ordonnées",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subtasks": {
                            "type": "array",
                            "description": "Liste des sous-tâches",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task": {
                                        "type": "string", 
                                        "description": "Description précise de la sous-tâche"
                                    },
                                    "agent": {
                                        "type": "string", 
                                        "description": "Agent recommandé pour la sous-tâche",
                                        "enum": [
                                            "Web Search Agent", 
                                            "API Knowledge Agent", 
                                            "Travel Planner Agent"
                                        ]
                                    },
                                    "priority": {
                                        "type": "string", 
                                        "description": "Priorité de la sous-tâche",
                                        "enum": ["haute", "moyenne", "basse"]
                                    }
                                },
                                "required": ["task", "agent", "priority"]
                            }
                        }
                    },
                    "required": ["subtasks"]
                }
            }
        ]

    def _get_agent_selection_functions(self) -> List[Dict[str, Any]]:
        """
        Définir les fonctions d'appel pour la sélection d'agents
        
        Args:
        
        Returns:
            List[Dict[str, Any]]: Liste des définitions de fonctions
        """
        return [
            {
                "name": "select_best_agent",
                "description": "Sélectionner l'agent le plus approprié pour une tâche donnée",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string", 
                            "description": "Description de la tâche à exécuter"
                        },
                        "selected_agent": {
                            "type": "object",
                            "description": "Agent sélectionné avec son score de confiance",
                            "properties": {
                                "name": {
                                    "type": "string", 
                                    "description": "Nom de l'agent",
                                    "enum": [
                                        "Web Search Agent", 
                                        #"API Knowledge Agent", 

                                    ]
                                },
                                "confidence_score": {
                                    "type": "number", 
                                    "description": "Score de confiance (0.0-1.0)",
                                    "minimum": 0,
                                    "maximum": 1
                                },
                                "reasoning": {
                                    "type": "string", 
                                    "description": "Justification de la sélection"
                                }
                            },
                            "required": ["name", "confidence_score", "reasoning"]
                        }
                    },
                    "required": ["task", "selected_agent"]
                }
            }
        ]

    def should_decompose_task(self, user_request: str) -> bool:
        """
        Utiliser le LLM pour déterminer si la tâche nécessite une décomposition
        
        Args:
            user_request (str): La requête utilisateur originale
        
        Returns:
            bool: True si décomposition nécessaire, False sinon
        """
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config["model"],
                messages=[
                    {
                        "role": "system", 
                        "content": """
                        Tu es un expert en analyse de tâches complexes.
                        
                        CRITÈRES DE DÉCOMPOSITION :
                        - La tâche nécessite-t-elle vraiment d'être divisée ?
                        - Présente-t-elle plusieurs dimensions ou étapes ?
                        - Requiert-elle différentes compétences ou approches ?
                        
                        NE PAS DÉCOMPOSER pour :
                        - Requêtes simples et directes
                        - Tâches ne nécessitant qu'une seule action
                        - Demandes courtes et précises
                        
                        DÉCOMPOSER pour :
                        - Projets multi-étapes
                        - Tâches nécessitant planification
                        - Requêtes complexes avec plusieurs objectifs
                        
                        Réponds par "OUI" ou "NON" avec discernement
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"Analyse cette requête : {user_request}"
                    }
                ],
                max_tokens=10,
                temperature=0.2
            )
            
            llm_decision = response.choices[0].message.content.strip().upper()
            logger.info(f"Décision de décomposition pour '{user_request}': {llm_decision}")
            
            return llm_decision == "OUI"
        
        except Exception as e:
            logger.error(f"Erreur lors de la décision de décomposition : {e}")
            return False

    def _select_best_agent(self, task: str) -> Agent:
        """
        Sélectionner dynamiquement le meilleur agent pour une tâche donnée
        
        Args:
            task (str): La tâche à exécuter
        
        Returns:
            Agent: L'agent le plus approprié
        """
        # Sélection par LLM
        try:
            # Utiliser l'API OpenAI pour classifier la tâche
            response = self.client.chat.completions.create(
                model=self.llm_config['model'],
                messages=[
                    {
                        "role": "system", 
                        "content": "Tu es un assistant qui aide à sélectionner le bon agent parmi une liste."
                    },
                    {
                        "role": "user", 
                        "content": f"""
                        Étant donné la tâche suivante, détermine quel agent serait le plus approprié :
                        
                        Tâche : {task}
                        
                        Agents disponibles : {list(self.agents.keys())}
                        
                        Réponds uniquement avec le nom de l'agent le plus adapté.
                        """
                    }
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            # Extraire et traiter la classification
            classification = response.choices[0].message.content.strip().lower()
            logger.info(f"🧠 Classification de la tâche : {classification}")
            logger.info(f"🔍 Agents disponibles : {list(self.agents.keys())}")

            # Convertir le nom en agent
            selected_agent_name = classification.lower().replace(' ', '_')
            selected_agent = self.agents.get(selected_agent_name)
            
            logger.info(f"🏆 Agent sélectionné : {selected_agent_name}")
            return selected_agent
        
        except Exception as e:
            logger.error(f"❌ Erreur de sélection d'agent : {e}")
            return next(iter(self.agents.values()))

    def _publish_rabbitmq_message(self, queue_name: str, message: Dict[str, Any]) -> bool:
        """
        Publier un message dans une file RabbitMQ
        
        Args:
            queue_name (str): Nom de la file
            message (Dict[str, Any]): Message à publier
        
        Returns:
            bool: True si la publication a réussi, False sinon
        """
        try:
            # Importer pika de manière sécurisée
            import importlib
            import os
            from dotenv import load_dotenv
            
            # Charger les variables d'environnement
            load_dotenv()
            
            # Récupérer les paramètres de connexion RabbitMQ
            rabbitmq_host = os.getenv('RABBITMQ_HOST', 'localhost')
            rabbitmq_port = int(os.getenv('RABBITMQ_PORT', 5672))
            rabbitmq_user = os.getenv('RABBITMQ_USER', '')
            rabbitmq_password = os.getenv('RABBITMQ_PASSWORD', '')
            
            # Importer pika
            pika = importlib.import_module('pika')
            
            # Configurer les paramètres de connexion
            credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_password) if rabbitmq_user else None
            connection_params = pika.ConnectionParameters(
                host=rabbitmq_host,
                port=rabbitmq_port,
                credentials=credentials,
                connection_attempts=2,
                retry_delay=1
            )
            
            # Établir la connexion
            try:
                connection = pika.BlockingConnection(connection_params)
            except (pika.exceptions.AMQPConnectionError, ConnectionRefusedError) as conn_error:
                logger.warning(f"Impossible de se connecter à RabbitMQ : {conn_error}")
                return False
            
            try:
                channel = connection.channel()
                
                # Déclarer la queue si elle n'existe pas
                channel.queue_declare(queue=queue_name, durable=True)
                
                # Publier le message
                channel.basic_publish(
                    exchange='',
                    routing_key=queue_name,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Rendre le message persistant
                        content_type='application/json'
                    )
                )
                
                logger.info(f"Message publié dans la file {queue_name} sur {rabbitmq_host}:{rabbitmq_port}")
                return True
            
            except Exception as publish_error:
                logger.error(f"Erreur lors de la publication dans RabbitMQ : {publish_error}")
                return False
            
            finally:
                connection.close()
        
        except ImportError:
            logger.warning("La bibliothèque pika n'est pas installée.")
            return False
        
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la publication RabbitMQ : {e}")
            return False

    async def decompose_task(self, user_request: str) -> TaskLedger:
        """
        Décomposer la requête utilisateur en sous-tâches avec function calling
        
        Args:
            user_request (str): La requête utilisateur originale
        
        Returns:
            TaskLedger: Le registre de tâches mis à jour
        """
        try:
            # Générer un ID unique pour la tâche
            task_id = str(uuid.uuid4())
            
            # Décomposer la tâche
            detailed_subtasks = self._generate_detailed_subtasks(user_request)
            
            # Mettre à jour le TaskLedger
            self.task_ledger.current_plan = [
                subtask['description']
                for subtask in detailed_subtasks
            ]
            
            # Préparer le message de progression pour RabbitMQ
            progress_message = {
                'task_id': task_id,
                'original_request': user_request,
                'total_subtasks': len(detailed_subtasks),
                'subtasks': detailed_subtasks,
                'status': 'started',
                'timestamp': datetime.now().isoformat()
            }
            
            # Publier le message dans la queue de progression
            self._publish_rabbitmq_message('queue_progress_task', progress_message)
            
            return self.task_ledger
        
        except Exception as e:
            logger.error(f"Erreur lors de la décomposition de tâche : {e}")
            # Retourner le TaskLedger même en cas d'erreur
            return self.task_ledger

    async def execute_task(
        self, 
        task_ledger: TaskLedger, 
        dev_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Exécuter les sous-tâches de manière unifiée
        
        Args:
            task_ledger (TaskLedger): Le registre de tâches à exécuter
            dev_mode (bool): Mode développement qui simule l'exécution
        
        Returns:
            Dict[str, Any]: Résultats de l'exécution des tâches
        """
        try:
            # Initialiser le dictionnaire des résultats
            task_results = {}
            
            # Exécuter chaque sous-tâche
            for task_index, task in enumerate(task_ledger.current_plan, 1):
                try:
                    # Sélectionner l'agent le plus approprié
                    selected_agent = self._select_best_agent(task)
                    
                    # Exécuter la sous-tâche de manière asynchrone
                    try:
                        # Utiliser arun() de manière asynchrone
                        result = await selected_agent.arun(task)
                    except AttributeError:
                        # Lever une exception si arun() n'est pas disponible
                        raise RuntimeError(f"L'agent {selected_agent.name} ne supporte pas l'exécution asynchrone")
                    
                    # Stocker le résultat
                    task_results[task] = {
                        "agent": selected_agent.name,
                        "result": {
                            "content": result.content,
                            "content_type": result.content_type,
                            "event": result.event,
                            "messages": [
                                {
                                    "role": msg.role, 
                                    "content": msg.content
                                } for msg in result.messages
                            ] if result.messages else []
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Publier le message dans la queue de progression avec le résultat
                    progress_message = {
                        "task_index": task_index,
                        "total_tasks": len(task_ledger.current_plan),
                        "task": task,
                        "agent": selected_agent.name,
                        "status": "completed",
                        "result": task_results[task]["result"],
                        "timestamp": datetime.now().isoformat()
                    }
                    self._publish_rabbitmq_message('queue_progress_task', progress_message)
                    
                    # Log de succès
                    logger.info(f"✅ Sous-tâche {task_index} terminée")
                    logger.info(f"📊 Résultat : {str(result)[:200]}...")
                    
                except Exception as task_error:
                    logger.error(f"❌ Erreur lors de l'exécution de la sous-tâche {task_index} : {task_error}")
                    logger.error(traceback.format_exc())
                    
                    task_results[task] = {
                        'result': f"Erreur : {str(task_error)}",
                        'agent': 'error',
                        'traceback': traceback.format_exc()
                    }
            
            # Synthétiser les résultats
            try:
                synthesized_result = await self._synthesize_results(list(task_results.values()))
                logger.info("🏁 Exécution de toutes les sous-tâches terminée")
                logger.info(f"📋 Résultat synthétisé : {str(synthesized_result)[:200]}...")
            except Exception as synthesis_error:
                logger.error(f"❌ Erreur lors de la synthèse : {synthesis_error}")
                synthesized_result = "Désolé, je n'ai pas pu synthétiser les résultats."
            
            return {
                'task_results': task_results,
                'synthesized_result': synthesized_result
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur globale lors de l'exécution des tâches : {e}")
            logger.error(traceback.format_exc())
            
            return {
                'task_results': {},
                'synthesized_result': "Erreur lors de l'exécution des tâches."
            }

    async def process_request(
        self, 
        user_request: str, 
        debug_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Traiter une requête de bout en bout
        
        Args:
            user_request (str): La requête utilisateur
            debug_mode (bool): Mode de débogage
        
        Returns:
            Dict[str, Any]: Résultats du traitement
        """
        try:
            # Décider si la tâche nécessite une décomposition
            needs_decomposition = self.should_decompose_task(user_request)
            logger.info(f"🔍 Décomposition requise : {needs_decomposition}")
            
            # Sélectionner le mode de traitement
            if needs_decomposition:
                # Décomposer la tâche en sous-tâches
                task_ledger = await self.decompose_task(user_request)
                
                # Exécuter les sous-tâches
                task_results = await self.execute_task(task_ledger)
                
                return task_results
            
            # Si pas de décomposition, exécuter directement
            selected_agent = self._select_best_agent(user_request)
            result = await selected_agent.arun(user_request)
            
            # Préparer le message RabbitMQ
            task_result_message = {
                "task_index": 1,
                "total_tasks": 1,
                "task": user_request,
                "agent": selected_agent.name,
                "status": "completed",
                "result": {
                    "content": result.content,
                    "content_type": result.content_type,
                    "event": result.event,
                    "messages": [
                        {
                            "role": msg.role, 
                            "content": msg.content
                        } for msg in result.messages
                    ] if result.messages else []
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Publier le message dans la queue de progression
            self._publish_rabbitmq_message('queue_progress_task', task_result_message)
            
            # Log détaillé
            logger.info(f"📊 Résultat complet : {result}")
            logger.info(f"📝 Contenu : {result.content}")
            
            return {
                'task_results': {user_request: result},
                'synthesized_result': result.content
            }
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la requête : {e}")
            logger.error(traceback.format_exc())
            
            return {
                'task_results': {},
                'synthesized_result': "Erreur lors du traitement de la requête."
            }

    async def _synthesize_results(
        self, 
        subtask_results: List[Union[RunResponse, Dict, str]]
    ) -> str:
        """
        Synthétiser les résultats de plusieurs sous-tâches
        
        Args:
            subtask_results (List[Union[RunResponse, Dict, str]]): Liste des résultats de sous-tâches
        
        Returns:
            str: Résultat synthétisé
        """
        logger.info("🏁 Début de la synthèse des résultats")
        
        # Convertir les résultats en format texte si nécessaire
        text_results = []
        for result in subtask_results:
            # Gérer différents types de résultats
            if isinstance(result, RunResponse):
                text_results.append(result.content)
            elif isinstance(result, dict):
                # Extraire le contenu du résultat
                content = result.get('result', {})
                if isinstance(content, dict):
                    text_results.append(content.get('content', str(content)))
                else:
                    text_results.append(str(content))
            else:
                text_results.append(str(result))
        
        # Cas spécial : résultat unique
        if len(text_results) == 1:
            return text_results[0]
        
        # Utiliser l'agent orchestrateur pour synthétiser
        synthesis_prompt = f"""
        Synthétise les résultats suivants de manière concise et claire :
        
        {chr(10).join(text_results)}
        
        Règles pour la synthèse :
        - Si plusieurs étapes, numérote et résume chaque étape
        - Fournis un résumé final qui capture l'essence de tous les résultats
        - Sois concis mais informatif
        """
        
        try:
            # Essayer d'utiliser arun() en premier
            synthesis_response = await self.agent.arun(synthesis_prompt)
            synthesized_result = synthesis_response.content
        except Exception:
            # Fallback à la méthode synchrone
            synthesis_response = self.agent(synthesis_prompt)
            synthesized_result = synthesis_response.choices[0].message.content.strip()
        
        # Publier un message RabbitMQ avec la synthèse
        try:
            synthesis_message = {
                "task_index": "synthesis",
                "total_tasks": len(subtask_results),
                "task": "Result Synthesis",
                "agent": "OrchestratorAgent",
                "status": "completed",
                "result": {
                    "content": synthesized_result,
                    "content_type": "text/plain",
                    "messages": []
                },
                "timestamp": datetime.now().isoformat()
            }
            self._publish_rabbitmq_message('queue_progress_task', synthesis_message)
        except Exception as e:
            logger.error(f"❌ Erreur lors de la publication de la synthèse : {e}")
        
        logger.info(f"📋 Résultat synthétisé : {synthesized_result}")
        return synthesized_result

    def _generate_detailed_subtasks(self, user_request: str) -> List[Dict[str, Any]]:
        """
        Générer des sous-tâches détaillées pour une requête utilisateur
        
        Args:
            user_request (str): La requête originale de l'utilisateur
        
        Returns:
            List[Dict[str, Any]]: Liste des sous-tâches détaillées
        """
        try:
            # Préparer le prompt pour la génération de sous-tâches
            subtasks_prompt = f"""
            Décompose la requête suivante en sous-tâches précises et réalisables :
            
            Requête : {user_request}
            
            Instructions :
            - Divise la tâche en étapes concrètes et mesurables
            - Chaque sous-tâche doit être claire et réalisable
            - Inclure des détails sur l'objectif de chaque sous-tâche
            - Estimer un temps approximatif pour chaque sous-tâche
            
            Format de réponse REQUIS (JSON strict) :
            {{
                "subtasks": [
                    {{
                        "task_id": "identifiant_unique",
                        "description": "Description détaillée de la sous-tâche",
                        "estimated_time": "Temps estimé",
                        "priority": "haute/moyenne/basse",
                        "required_skills": ["compétences requises"]
                    }}
                ]
            }}
            """
            
            # Générer les sous-tâches directement avec le client OpenAI
            response = self.client.chat.completions.create(
                model=self.llm_config.get('model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": "Tu es un expert en décomposition de tâches complexes."},
                    {"role": "user", "content": subtasks_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Extraire le contenu de la réponse
            subtasks_response = response.choices[0].message.content
            
            # Vérifier que la réponse est une chaîne
            if not isinstance(subtasks_response, str):
                logger.error(f"La réponse du modèle n'est pas une chaîne : {type(subtasks_response)}")
                raise ValueError("Réponse du modèle invalide")
            
            # Convertir la réponse en liste de sous-tâches
            try:
                subtasks_dict = json.loads(subtasks_response)
            except json.JSONDecodeError as json_err:
                logger.error(f"Erreur de décodage JSON : {json_err}")
                logger.error(f"Réponse reçue : {subtasks_response}")
                raise
            
            subtasks = subtasks_dict.get('subtasks', [])
            
            # Vérifier que le format est correct
            if not isinstance(subtasks, list):
                raise ValueError("La réponse n'est pas une liste de sous-tâches")

            return subtasks
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération des sous-tâches : {e}")
            # Retourner une décomposition par défaut si la génération échoue
            return [
                {
                    "task_id": "task_1",
                    "description": f"Analyser la requête : {user_request}",
                    "estimated_time": "1 heure",
                    "priority": "haute",
                    "required_skills": ["compréhension", "analyse"]
                }
            ]

async def process_user_request(
    user_request: str, 
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Traiter une requête utilisateur de manière asynchrone avec l'orchestrateur
    
    Args:
        user_request (str): La requête de l'utilisateur
        debug_mode (bool): Mode de débogage
    
    Returns:
        Dict[str, Any]: Résultat du traitement de la requête
    """
    try:
        orchestrator = OrchestratorAgent(
            debug_mode=debug_mode, 
            original_request=user_request
        )
        result = await orchestrator.process_request(
            user_request=user_request,
            debug_mode=debug_mode
        )
        
        # Extraction de la synthèse
        synthesized_result = result.get('synthesized_result', '')
        
        # Détermination de l'agent utilisé
        agent_used = 'Multi-Purpose Intelligence Team'
        task_results = result.get('task_results', {})
        
        # Cas avec plusieurs tâches : utiliser la synthèse
        if len(task_results) > 1:
            agent_used = list(task_results.values())[0].get('agent', agent_used)
        
        # Cas avec une seule tâche : extraire le contenu
        elif len(task_results) == 1:
            first_task = list(task_results.keys())[0]
            task_result = task_results[first_task]
            
            # Si c'est un RunResponse
            if hasattr(task_result, 'content'):
                synthesized_result = task_result.content
                agent_used = task_result.name if hasattr(task_result, 'name') else agent_used
            
            # Si c'est un dictionnaire
            elif isinstance(task_result, dict):
                # Essayer d'extraire le contenu de différentes manières
                if 'result' in task_result:
                    result_content = task_result['result']
                    if isinstance(result_content, dict) and 'content' in result_content:
                        synthesized_result = result_content['content']
                    elif isinstance(result_content, str):
                        synthesized_result = result_content
                
                agent_used = task_result.get('agent', agent_used)
        
        return {
            'query': user_request,
            'result': synthesized_result,
            'agent_used': agent_used,
            'metadata': {},
            'error': None
        }
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête : {e}")
        logger.error(traceback.format_exc())
        return {
            'query': user_request,
            'result': '',
            'agent_used': 'Multi-Purpose Intelligence Team',
            'metadata': {},
            'error': str(e)
        }

def get_orchestrator_agent(
    model_id: str = "gpt-4o-mini",
    enable_web_agent: bool = True,
    enable_api_knowledge_agent: bool = False,
    enable_data_analysis_agent: bool = False,
    enable_travel_planner: bool = False
) -> OrchestratorAgent:
    """
    Créer un agent orchestrateur avec configuration personnalisable
    
    Args:
        model_id (str): Identifiant du modèle OpenAI
        enable_web_agent (bool): Activer l'agent de recherche web
        enable_api_knowledge_agent (bool): Activer l'agent de connaissances API
        enable_data_analysis_agent (bool): Activer l'agent d'analyse de données
        enable_travel_planner (bool): Activer l'agent de planification de voyage
    
    Returns:
        OrchestratorAgent: Agent orchestrateur configuré
    """
    return OrchestratorAgent(
        model_id=model_id,
        enable_web_agent=enable_web_agent,
        enable_api_knowledge_agent=enable_api_knowledge_agent,
        enable_data_analysis_agent=enable_data_analysis_agent,
        enable_travel_planner=enable_travel_planner
    )

# Exemple d'utilisation
if __name__ == "__main__":
    import logging
    import json

    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    def extract_content(chunk):
        """
        Extraire le contenu textuel d'un chunk de différents types
        """
        # Si c'est un tuple
        if isinstance(chunk, tuple):
            # Si le tuple a plus d'un élément, prendre le deuxième
            if len(chunk) > 1:
                chunk = chunk[1]
            else:
                chunk = ""
        
        # Si c'est une liste, convertir en chaîne
        if isinstance(chunk, list):
            chunk = " ".join(str(item) for item in chunk)
        
        # Convertir en chaîne si ce n'est pas déjà une chaîne
        return str(chunk)

    def test_orchestrator():
        # Créer l'agent orchestrateur sans agent web
        orchestrator = get_orchestrator_agent(
            enable_web_agent=True  # Activer la recherche web
        )
        
        # Exemples de requêtes de test
        test_requests = [
            "Faire une analyse comparative des performances des startups tech en 2024"
        ]
        
        for request in test_requests:
            print(f"\n🚀 Traitement de la requête : {request}")
            
            # Utiliser la génération de réponse directe
            try:
                # Récupérer le générateur de réponse
                resp = orchestrator.run(request)
                
                # Collecter et afficher les résultats par morceaux
                print("\n📊 Résultats :")
                full_result = ""
                for chunk in resp:
                    # Extraire le contenu du chunk
                    chunk_content = extract_content(chunk)
                    
                    print(chunk_content, end='', flush=True)
                    full_result += chunk_content
                
                print("\n\n🔍 Résumé :")
                print(f"Longueur totale de la réponse : {len(full_result)} caractères")
            except Exception as e:
                print(f"Erreur lors de l'exécution : {e}")
                import traceback
                traceback.print_exc()
    
    # Exécuter le test
    test_orchestrator()