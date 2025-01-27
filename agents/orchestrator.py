import os
import sys
import asyncio
import logging
import traceback
import uuid
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import time

import openai
from phi.agent import RunResponse, Agent, AgentMemory
from phi.model.openai import OpenAIChat
from phi.memory.db.postgres import PgMemoryDb
from phi.storage.agent.postgres import PgAgentStorage
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.memory.db.sqlite import SqliteMemoryDb

from agents.web import get_web_searcher
from agents.settings import agent_settings
from agents.orchestrator_prompts import (
    get_task_decomposition_prompt,
    get_task_execution_prompt,
    get_task_synthesis_prompt
)

from utils.colored_logging import get_colored_logger



agent_memory_file: str = "orchestrator_agent_memory.db"
agent_storage_file: str = "orchestrator_agent_sessions.db"


# Configuration du logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ajout d'un handler de console si nécessaire
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Ajouter le répertoire parent au PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

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
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
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
            "task_id": self.task_id,
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

class OrchestratorAgent(Agent):
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
        enable_travel_planner: bool = False,
        # Ajout des paramètres de la classe Agent
        name: str = "Orchestrator Agent",
        instructions: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        storage: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialiser l'agent orchestrateur avec des agents spécialisés et une mémoire persistante
        
        Args:
            model_id (str): Identifiant du modèle OpenAI
            debug_mode (bool): Mode de débogage
            original_request (Optional[str]): Requête originale pour le TaskLedger
            api_key (Optional[str]): Clé API OpenAI personnalisée
            enable_web_agent (bool): Activer l'agent de recherche web
            enable_api_knowledge_agent (bool): Activer l'agent de connaissances API
            enable_data_analysis_agent (bool): Activer l'agent d'analyse de données
            enable_travel_planner (bool): Activer l'agent de planification de voyage
            db_url (str): URL de connexion à la base de données PostgreSQL
            memory_table_name (str): Nom de la table de mémoire
            storage_table_name (str): Nom de la table de stockage
        """
        # Préparer les instructions par défaut si non fournies
        default_instructions = [
            "Tu es un agent d'orchestration intelligent.",
            "Décompose les tâches complexes en sous-tâches gérables.",
            "Sélectionne et coordonne les agents spécialisés.",
            "Assure une exécution efficace et cohérente des tâches."
        ]
        instructions = kwargs.get('instructions', default_instructions)
        # Configuration du stockage
        storage = SqlAgentStorage(
            table_name="agent_sessions", 
            db_file=agent_storage_file
        )

        # Préparation des paramètres pour l'initialisation
        agent_init_kwargs = {
            "name": "Orchestrator Agent",
            "instructions": instructions,
            "tools": tools,
            "user_id": user_id,
            "session_id": session_id,
            "memory": AgentMemory(
                # db=PgMemoryDb(
                #     table_name=kwargs.get('memory_table_name', 'orchestrator_agent_memory'), 
                #     db_url=kwargs.get('db_url', 'postgresql+psycopg2://p4t:o3CCgX7StraZqvRH5GqrOFLuzt5R6C@vps-af24e24d.vps.ovh.net:30030/myboun')
                # ),
                    db=SqliteMemoryDb(
                        table_name="agent_memory",
                        db_file=agent_memory_file,
                    ),
            # Create and store personalized memories for this user
            create_user_memories=True,
            # Update memories for the user after each run
            update_user_memories_after_run=True,
            # Create and store session summaries
            create_session_summary=True,
            # Update session summaries after each run
            update_session_summary_after_run=True,
            ),
            "storage": storage,
            "add_history_to_messages": True,
            "num_history_responses": 3
        }
        
        # Ajouter les kwargs supplémentaires
        agent_init_kwargs.update(kwargs)

        # Appel du constructeur parent de Agent
        super().__init__(
            name="Orchestrator Agent",
            instructions=instructions,
            tools=tools,
            user_id=user_id,
            session_id=session_id,
            memory=memory,
            storage=storage,
            **kwargs
        )
        
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

    def run(self, task: str) -> RunResponse:
        """
        Méthode run standard pour l'Agent Phidata
        Délègue au processus de traitement de requête existant
        """
        try:
            result = self.process_request(task, debug_mode=self.debug_mode)
            return RunResponse(
                content=result.get('final_result', 'Aucun résultat'),
                content_type='text',
                metadata=result
            )
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution de l'orchestrateur : {e}")
            return RunResponse(
                content=f"Erreur : {str(e)}",
                content_type='error'
            )

    def _initialize_specialized_agents(
        self, 
        enable_web_agent: bool = True,
        enable_api_knowledge_agent: bool = False,
        enable_data_analysis_agent: bool = False,
        enable_travel_planner: bool = False
    ) -> Dict[str, Any]:
        """
        Initialiser les agents spécialisés
        
        Args:
            enable_web_agent (bool): Activer l'agent de recherche web
            enable_api_knowledge_agent (bool): Activer l'agent de connaissances API
            enable_data_analysis_agent (bool): Activer l'agent d'analyse de données
            enable_travel_planner (bool): Activer l'agent de planification de voyage
        
        Returns:
            Dict[str, Any]: Dictionnaire des agents initialisés
        """
        logger.info("🤖 Initialisation des agents spécialisés")
        logger.info(f"🌐 Web Agent: {enable_web_agent}")
        logger.info(f"📚 API Knowledge Agent: {enable_api_knowledge_agent}")
        logger.info(f"📊 Data Analysis Agent: {enable_data_analysis_agent}")
        logger.info(f"✈️ Travel Planner Agent: {enable_travel_planner}")
        
        agents = {}
        
        # Initialisation de l'agent de recherche web
        if enable_web_agent:
            try:
                web_agent = get_web_searcher()
                agents['web_agent'] = web_agent
                logger.info("✅ Agent de recherche web initialisé avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur d'initialisation de l'agent de recherche web : {e}")
                logger.debug(traceback.format_exc())
        
        # Placeholder pour les autres agents (à implémenter si nécessaire)
        if enable_api_knowledge_agent:
            logger.warning("🚧 API Knowledge Agent non implémenté")
        
        if enable_data_analysis_agent:
            logger.warning("🚧 Data Analysis Agent non implémenté")
        
        if enable_travel_planner:
            logger.warning("🚧 Travel Planner Agent non implémenté")
        
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
                        Analyse la tâche suivante et choisis l'agent le plus adapté en fonction de ses compétences :

                        Tâche : {task}
                        
                        Liste des agents disponibles : {list(self.agents.keys())}

                        Critères de sélection :
                        - Compatibilité des compétences de l'agent avec la tâche.
                        - Spécialisation et capacité d'exécution de l'agent.
                        - Rapidité et efficacité de l'agent pour accomplir la tâche.
                        
                        Réponds uniquement avec le nom de l'agent le plus adapté.
                        """
                    }
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            # Extraire et traiter la classification
            classification = response.choices[0].message.content.strip().lower()
            logger.debug(f"🧠 Classification de la tâche : {classification}")
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
                
                logger.info(f"Message RabbitMQ publié dans la file {queue_name} sur {rabbitmq_host}:{rabbitmq_port}")
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

    def _create_task_message(
        self, 
        task_type: str, 
        request_id: str, 
        sub_task_id: Optional[str] = None,
        original_request: Optional[str] = None,
        description: Optional[str] = None,
        subtasks: Optional[List[Dict[str, Any]]] = None,
        result: Optional[Dict[str, Any]] = None,
        status: str = 'pending'
    ) -> Dict[str, Any]:
        """
        Créer un message standardisé pour les tâches, sous-tâches et synthèse.
        
        Args:
            task_type (str): Type de tâche ('task', 'subtask', 'synthesis')
            request_id (str): ID unique de la demande
            sub_task_id (str, optional): ID de la sous-tâche
            original_request (str, optional): Requête originale
            description (str, optional): Description de la tâche
            subtasks (List[Dict], optional): Liste des sous-tâches
            result (Dict, optional): Résultat de la tâche
            status (str, optional): Statut de la tâche
        
        Returns:
            Dict[str, Any]: Message standardisé
        """
        message = {
            'message_type': 'task_progress',
            'task_type': task_type,
            'request_id': request_id,
            'sub_task_id': sub_task_id,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'original_request': original_request,
                'description': description
            }
        }
        
        # Ajouter des informations spécifiques selon le type de tâche
        if subtasks:
            message['subtasks'] = [
                {
                    'sub_task_id': subtask.get('sub_task_id', str(uuid.uuid4())),
                    'description': subtask.get('description'),
                    'status': subtask.get('status', 'pending')
                } 
                for subtask in subtasks
            ]
            message['total_subtasks'] = len(subtasks)
        
        if result:
            message['result'] = result
        
        return message

    async def decompose_task(self, user_request: str) -> TaskLedger:
        """
        Décomposer la requête utilisateur en sous-tâches avec function calling
        
        Args:
            user_request (str): La requête utilisateur originale
        
        Returns:
            TaskLedger: Le registre de tâches mis à jour
        """
        logger.info(f"🧩 Décomposition de la tâche principale : {user_request}")
        
        try:
            # Décomposer la tâche
            detailed_subtasks = self._generate_detailed_subtasks(user_request)
            
            # Ajouter des sub_task_id aux sous-tâches
            for subtask in detailed_subtasks:
                subtask['sub_task_id'] = str(uuid.uuid4())
            
            # Mettre à jour le TaskLedger
            self.task_ledger.current_plan = [
                subtask['description']
                for subtask in detailed_subtasks
            ]
            
            # Log détaillé des sous-tâches identifiées
            logger.info(f"📋 Nombre de sous-tâches identifiées : {len(self.task_ledger.current_plan)}")
            for idx, subtask in enumerate(self.task_ledger.current_plan, 1):
                logger.info(f"🔢 Sous-tâche {idx}: {subtask}")
            
            # Préparer le message de tâche principal
            task_message = self._create_task_message(
                task_type='task',
                request_id=self.task_ledger.task_id,
                original_request=user_request,
                subtasks=detailed_subtasks,
                status='started'
            )
            
            # Publier le message dans la queue de progression
            self._publish_rabbitmq_message('queue_progress_task', task_message)
            
            return self.task_ledger
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la décomposition de tâche : {e}")
            logger.error(f"🔍 Trace complète : {traceback.format_exc()}")
            
            # En cas d'erreur, retourner le TaskLedger avec la tâche originale
            self.task_ledger.current_plan = [user_request]
            return self.task_ledger

    async def execute_task(
        self, 
        task_ledger: TaskLedger, 
        dev_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Exécuter les sous-tâches de manière unifiée
        
        Args:
            task_ledger (TaskLedger): Le registre de tâches à exécuter
            dev_mode (bool): Mode développement qui simule l'exécution
        
        Returns:
            List[Dict[str, Any]]: Résultats des sous-tâches
        """
        subtask_results = []
        
        try:
            # Parcourir les sous-tâches du registre
            for task_index, task in enumerate(task_ledger.current_plan, 1):
                logger.info(f"🔢 Sous-tâche {task_index}/{len(task_ledger.current_plan)}: {task}")
                
                # Sélectionner dynamiquement l'agent
                selected_agent = self._select_best_agent(task)
                
                # Générer un ID unique pour cette sous-tâche
                sub_task_id = str(uuid.uuid4())
                
                # Exécuter la sous-tâche avec vérification des méthodes disponibles de l'agent
                logger.info(f"🔍 Exécution de la tâche : {task}")
                
                # Récupérer le nom de l'agent de plusieurs manières
                agent_name = (
                    getattr(selected_agent, 'name', None) or  # Attribut 'name' défini lors de la création
                    getattr(selected_agent, '__name__', None) or  # Nom de la classe
                    selected_agent.__class__.__name__  # Nom de la classe par défaut
                )
                logger.info(f"🤖 Réalisation de la sous-tâche par : {agent_name}")

                try:
                    start_time = time.time()
                    
                    if hasattr(selected_agent, 'run'):
                        logger.debug("📡 Utilisation de la méthode synchrone run()")
                        result = selected_agent.run(task)
                        logger.debug(f"✅ Méthode run() exécutée avec succès pour {selected_agent.__class__.__name__}")
                    
                    elif hasattr(selected_agent, 'arun'):
                        logger.debug("📡 Utilisation de la méthode asynchrone arun()")
                        result = await selected_agent.arun(task)
                        logger.debug(f"✅ Méthode arun() exécutée avec succès pour {selected_agent.__class__.__name__}")
                    
                    else:
                        logger.warning("⚠️ Aucune méthode run() ou arun() trouvée, utilisation du modèle LLM direct")
                        result = selected_agent.model(task)
                        logger.debug(f"✅ Modèle LLM utilisé pour {selected_agent.__class__.__name__}")

                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Log détaillé du résultat de la sous-tâche
                    logger.info(f"✨ Résultat de la sous-tâche : {result.content[:200]}...")
                    logger.info(f"⏱️ Temps d'exécution : {execution_time:.2f} secondes")
                    

                except Exception as e:
                    logger.error(f"❌ Erreur lors de l'exécution de l'agent {selected_agent.__class__.__name__}")
                    logger.error(f"🔴 Détails de l'erreur : {str(e)}")
                    logger.error(f"🔍 Trace complète : {traceback.format_exc()}")
                    result = None
                
                # Préparer le message de résultat de sous-tâche
                subtask_result_message = self._create_task_message(
                    task_type='subtask',
                    request_id=task_ledger.task_id,
                    sub_task_id=sub_task_id,
                    original_request=task,
                    status='completed',
                    result={
                        "content": result.content if result else "Aucun résultat",
                        "content_type": result.content_type if result else "error",
                        "agent": selected_agent.name
                    }
                )
                
                # Publier le message de résultat de sous-tâche
                self._publish_rabbitmq_message('queue_progress_task', subtask_result_message)
                
                # Stocker le résultat
                subtask_results.append({
                    'result': result.content if result else "Aucun résultat",
                    'agent': selected_agent.name
                })
    
            return subtask_results
    
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution des tâches : {e}")
            return []

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
            # Décomposer la tâche
            task_ledger = await self.decompose_task(user_request)
            
            # Exécuter les sous-tâches
            subtask_results = await self.execute_task(task_ledger)
            
            # Synthétiser les résultats
            synthesized_result = await self._synthesize_results(subtask_results)
            
            # Publier un message RabbitMQ avec la synthèse
            synthesis_message = self._create_task_message(
                task_type='synthesis',
                request_id=task_ledger.task_id,
                original_request=user_request,
                status='completed',
                result={
                    "content": synthesized_result,
                    "content_type": "text/plain"
                }
            )
            
            # Publier le message de synthèse
            self._publish_rabbitmq_message('queue_progress_task', synthesis_message)
            
            # Log détaillé
            logger.info(f"📊 Résultat synthétisé : {synthesized_result}")
            
            return {
                'query': user_request,
                'result': synthesized_result,
                'agent_used': 'Multi-Purpose Intelligence Team',
                'metadata': {},
                'error': None,
                'task_id': task_ledger.task_id
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de la requête : {e}")
            return {
                'query': user_request,
                'result': '',
                'agent_used': 'Multi-Purpose Intelligence Team',
                'metadata': {},
                'error': str(e),
                'task_id': None
            }

    async def _synthesize_results(
        self, 
        subtask_results: List[Dict[str, Any]]
    ) -> str:
        """
        Synthétiser les résultats de plusieurs sous-tâches
        
        Args:
            subtask_results (List[Dict[str, Any]]): Liste des résultats de sous-tâches
        
        Returns:
            str: Résultat synthétisé
        """
        logger.info("🏁 Début de la synthèse des résultats")
        
        # Convertir les résultats en format texte
        text_results = [
            result.get('result', '') 
            for result in subtask_results 
            if result.get('result')
        ]
        
        # Cas spécial : résultat unique
        if len(text_results) == 1:
            return text_results[0]
        
        # Cas où aucun résultat n'est disponible
        if not text_results:
            return "Aucun résultat n'a pu être généré."
        
        # Utiliser l'agent orchestrateur pour synthétiser
        synthesis_prompt = f"""
        Synthétise les résultats suivants de manière concise et claire :
        
        {chr(10).join(text_results)}
        
        Règles pour la synthèse :
        - Si plusieurs étapes, numérote et résume chaque étape
        - Fournis un résumé final qui capture l'essence de tous les résultats
        - Sois concis mais informatif
        """
        
        # Utiliser le modèle pour générer la synthèse
        try:
            synthesis_response = await self.agent.arun(synthesis_prompt)
            return synthesis_response.content
        except Exception as e:
            logger.error(f"❌ Erreur lors de la synthèse : {e}")
            # Retourner une synthèse par défaut en cas d'erreur
            return " | ".join(text_results)

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
            subtasks_prompt = """
            Décompose la tâche suivante en sous-tâches essentielles et non redondantes.
            
            Tâche principale : {user_request}
            
            Instructions:
            1. Analyse la tâche en détail
            2. Identifie les actions concrètes nécessaires
            3. Évite les étapes redondantes de rapport de résultat
            4. Concentre-toi sur les actions productives
            
            Format de réponse REQUIS (JSON strict) :
            {{
                "subtasks": [
                    {{
                        "task_id": "identifiant_unique",
                        "description": "Description concise et précise de la sous-tâche",
                        "priority": "haute|moyenne|basse"                    
                    }}
                ]
            }}
            """.format(user_request=user_request)
            
            # Générer les sous-tâches directement avec le client OpenAI
            response = self.client.chat.completions.create(
                model=self.llm_config.get('model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": "Tu es un expert en décomposition de tâches complexes, privilégiant la concision et l'efficacité."},
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
                    "description": f"Analyser et exécuter : {user_request}",
                    "priority": "haute"
                }
            ]

async def process_user_request(
    user_request: str, 
    debug_mode: bool = False,
    # Ajout des paramètres de mémoire et stockage
    db_url: str = 'postgresql+psycopg2://p4t:o3CCgX7StraZqvRH5GqrOFLuzt5R6C@vps-af24e24d.vps.ovh.net:30030/myboun',
    memory_table_name: str = "agent_memory",
    storage_table_name: str = "agent_sessions",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Traiter une requête utilisateur de manière asynchrone avec l'orchestrateur
    
    Args:
        user_request (str): La requête de l'utilisateur
        debug_mode (bool): Mode de débogage
        db_url (str): URL de connexion à la base de données PostgreSQL
        memory_table_name (str): Nom de la table de mémoire
        storage_table_name (str): Nom de la table de stockage
        user_id (Optional[str]): Identifiant utilisateur
        session_id (Optional[str]): Identifiant de session
    
    Returns:
        Dict[str, Any]: Résultat du traitement de la requête
    """
    try:
        orchestrator = OrchestratorAgent(
            debug_mode=debug_mode, 
            original_request=user_request,
            # Ajout des paramètres de mémoire et stockage
            db_url=db_url,
            memory_table_name=memory_table_name,
            storage_table_name=storage_table_name,
            user_id=user_id,
            session_id=session_id
        )
        result = await orchestrator.process_request(
            user_request=user_request,
            debug_mode=debug_mode
        )
        
        # Extraction de la synthèse
        synthesized_result = result.get('result', '')
        
        # Détermination de l'agent utilisé
        agent_used = 'Multi-Purpose Intelligence Team'
        task_results = result.get('task_results', {})
        
        # Cas spécial : résultat unique
        if len(task_results) > 1:
            agent_used = list(task_results.values())[0].get('agent', agent_used)
        
        # Cas où aucun résultat n'est disponible
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
                    if isinstance(result_content, dict):
                        synthesized_result = result_content.get('content', str(result_content))
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
    enable_travel_planner: bool = False,
    db_url: str = 'postgresql+psycopg2://p4t:o3CCgX7StraZqvRH5GqrOFLuzt5R6C@vps-af24e24d.vps.ovh.net:30030/myboun',
    memory_table_name: str = "agent_memory",
    storage_table_name: str = "agent_sessions",
    # Paramètres optionnels de l'agent
    name: str = "Orchestrator Agent",
    instructions: Optional[List[str]] = None,
    tools: Optional[List[Callable]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> OrchestratorAgent:
    """
    Créer un agent orchestrateur avec configuration personnalisable
    
    Args:
        model_id (str): Identifiant du modèle OpenAI
        enable_web_agent (bool): Activer l'agent de recherche web
        enable_api_knowledge_agent (bool): Activer l'agent de connaissances API
        enable_data_analysis_agent (bool): Activer l'agent d'analyse de données
        enable_travel_planner (bool): Activer l'agent de planification de voyage
        db_url (str): URL de connexion à la base de données PostgreSQL
        memory_table_name (str): Nom de la table de mémoire
        storage_table_name (str): Nom de la table de stockage
    
    Returns:
        OrchestratorAgent: Agent orchestrateur configuré
    """
    return OrchestratorAgent(
        model_id=model_id,
        enable_web_agent=enable_web_agent,
        enable_api_knowledge_agent=enable_api_knowledge_agent,
        enable_data_analysis_agent=enable_data_analysis_agent,
        enable_travel_planner=enable_travel_planner,
        # Paramètres de mémoire et stockage
        db_url=db_url,
        memory_table_name=memory_table_name,
        storage_table_name=storage_table_name,
        # Paramètres optionnels de l'agent
        name=name,
        instructions=instructions,
        tools=tools,
        user_id=user_id,
        session_id=session_id,
        **kwargs
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