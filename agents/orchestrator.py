from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
import traceback

import os
import openai
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.python import PythonTools
from phi.tools.duckduckgo import DuckDuckGo

from agents.settings import agent_settings
from agents.web import get_web_searcher
from agents.api_knowledge import get_api_knowledge_agent
from agents.data_analysis import get_data_analysis_agent
from agents.travel_planner import get_travel_planner
from agents.orchestrator_prompts import (
    get_task_decomposition_prompt,
    get_task_execution_prompt,
    get_task_synthesis_prompt
)

from utils.colored_logging import get_colored_logger

import asyncio
from .agent_registry import agent_registry, AgentMetadata

logger = get_colored_logger('agents.orchestrator', 'OrchestratorAgent')

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
        api_key: Optional[str] = None  # Ajout du paramètre api_key
    ):
        """
        Initialiser l'agent orchestrateur avec des agents spécialisés
        
        Args:
            model_id (str): Identifiant du modèle OpenAI
            debug_mode (bool): Mode de débogage
            original_request (Optional[str]): Requête originale pour le TaskLedger
            api_key (Optional[str]): Clé API OpenAI personnalisée
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
        
        # Initialisation centralisée des agents
        self.agents = self._initialize_specialized_agents()
        
        # Initialisation du mode de débogage
        self.debug_mode = debug_mode
        
        # Création du TaskLedger initial
        self.task_ledger = self._create_task_ledger(original_request)
        
        # Créer l'agent orchestrateur avec configuration simplifiée
        self.agent = self._create_orchestrator_agent(debug_mode)

    def _initialize_specialized_agents(self) -> Dict[str, Agent]:
        """
        Initialiser tous les agents spécialisés de manière centralisée
        
        Returns:
            Dict[str, Agent]: Dictionnaire des agents disponibles
        """
        return {
            "web_search": get_web_searcher(
                model_id=agent_settings.gpt_4,
                debug_mode=False,
                name="Web Search Agent"
            ),
            "api_knowledge": get_api_knowledge_agent(
                debug_mode=False,
                user_id=None,
                session_id=None
            ),
            "travel_planner": get_travel_planner(
                model_id=agent_settings.gpt_4,
                debug_mode=False,
                name="Travel Planner Agent"
            )
        }

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

    def _create_orchestrator_agent(self, debug_mode: bool) -> Agent:
        """
        Créer l'agent orchestrateur avec configuration unifiée
        
        Args:
            debug_mode (bool): Mode de débogage
        
        Returns:
            Agent: Agent orchestrateur configuré
        """
        return Agent(
            llm=OpenAIChat(**self.llm_config),
            tools=[
                PythonTools(),
                DuckDuckGo()
            ],
            instructions=[
                "Tu es un agent d'orchestration avancé capable de décomposer des tâches complexes.",
                "Étapes de travail :",
                "1. Analyser la requête initiale",
                "2. Décomposer en sous-tâches précises",
                "3. Sélectionner l'agent le plus approprié",
                "4. Coordonner l'exécution des sous-tâches",
                "5. Intégrer et synthétiser les résultats partiels",
                "6. Adapter dynamiquement le plan si nécessaire"
            ],
            team=list(self.agents.values()),
            debug_mode=debug_mode,
            name="Advanced Task Orchestrator"
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
                                        "API Knowledge Agent", 
                                        "Travel Planner Agent"
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
        Sélectionner l'agent le plus approprié avec function calling
        
        Args:
            task (str): Tâche à exécuter
        
        Returns:
            Agent: Agent sélectionné
        """
        try:
            # Vérifier si le client est initialisé
            if self.client is None:
                logger.warning("Client OpenAI non initialisé. Utilisation de l'agent par défaut.")
                return next(iter(self.agents.values()))

            # Utiliser le function calling pour la sélection d'agent
            messages = [
                {"role": "system", "content": "Tu es un agent d'orchestration capable de sélectionner l'agent le plus approprié pour une tâche."},
                {"role": "user", "content": f"Sélectionne l'agent le plus approprié pour la tâche : {task}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.llm_config.get('model', 'gpt-4o-mini'),
                messages=messages,
                functions=self._get_agent_selection_functions(),
                function_call={"name": "select_best_agent"}
            )
            
            # Extraire l'agent sélectionné
            function_call = response.choices[0].message.function_call
            selection_data = json.loads(function_call.arguments)
            selected_agent_name = selection_data.get('selected_agent', {}).get('name', 'Web Search Agent')
            
            # Convertir le nom en agent
            selected_agent = self.agents.get(selected_agent_name.lower().replace(' ', '_'), 
                                             next(iter(self.agents.values())))
            
            return selected_agent
        
        except Exception as e:
            logger.warning(f"Erreur de sélection d'agent : {e}. Utilisation de l'agent par défaut.")
            return next(iter(self.agents.values()))

    def _parse_json_response(self, response: Any) -> List[Dict[str, str]]:
        """
        Parser une réponse JSON de manière robuste
        
        Args:
            response (Any): La réponse à parser
        
        Returns:
            List[Dict[str, str]]: Liste des sous-tâches parsées
        """
        try:
            # Extraction du function call
            function_call = response.choices[0].message.function_call
            subtasks_data = json.loads(function_call.arguments)
            subtasks = subtasks_data.get('subtasks', [])
            
            # Vérification et enrichissement des sous-tâches
            if not subtasks:
                # Génération de sous-tâches détaillées si vide
                subtasks = self._generate_detailed_subtasks(self.task_ledger.original_request)
            
            # Validation et complétion des sous-tâches
            validated_subtasks = []
            for subtask in subtasks:
                validated_task = {
                    'task': subtask.get('task', 'Tâche non spécifiée'),
                    'agent': subtask.get('agent', 'Web Search Agent'),
                    'priority': subtask.get('priority', 'moyenne')
                }
                validated_subtasks.append(validated_task)
            
            return validated_subtasks
        
        except Exception as e:
            logger.warning(f"Erreur de parsing JSON : {e}")
            # Génération de sous-tâches par défaut
            return self._generate_detailed_subtasks(self.task_ledger.original_request)

    def _generate_detailed_subtasks(self, request: str) -> List[Dict[str, str]]:
        """
        Générer des sous-tâches détaillées basées sur la requête
        """
        try:
            # Première étape : identification de l'objectif global
            global_objective_response = self.client.chat.completions.create(
                model=self.llm_config["model"],
                messages=[
                    {
                        "role": "system",
                        "content": """
                        Tu es un expert en analyse stratégique et définition d'objectifs.
                        
                        INSTRUCTIONS CRUCIALES :
                        1. Analyse la requête de manière NEUTRE et OBJECTIVE
                        
                        2. Si la requête contient des références temporelles relatives (ex: 'demain', 'dans une semaine', 'hier'):
                           - Utilise la date actuelle ({current_date}) comme référence
                           - Convertis ces références en dates précises avant d'effectuer la recherche
                           - Exemple: 'événements de demain' -> 'événements du {tomorrow_date}'
                        
                        3. Extraire l'OBJECTIF GLOBAL d'une requête
                           - Être concis et précis
                           - Capturer l'essence de la demande
                        
                        FORMAT :
                        {
                            "global_objective": "Description claire et concise",
                            "objective_type": "résolution/exploration/planification/...",
                            "key_dimensions": ["dimension1", "dimension2"],
                            "initial_constraints": ["contrainte1", "contrainte2"]
                        }
                        """.format(
                            current_date=datetime.now().strftime("%Y-%m-%d"),
                            tomorrow_date=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                        )
                    },
                    {
                        "role": "user",
                        "content": request
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=0.3
            )
            
            global_objective_data = json.loads(global_objective_response.choices[0].message.content)
            
            # Deuxième étape : génération des sous-tâches avec résultats cumulatifs
            response = self.client.chat.completions.create(
                model=self.llm_config["model"],
                messages=[
                    {
                        "role": "system", 
                        "content": """
                        Tu es un expert en décomposition de tâches séquentielles avec une stratégie de réduction de périmètre.

                        PRINCIPES FONDAMENTAUX :
                        1. ORDONNANCEMENT STRATEGIQUE des sous-tâches
                           - Chaque sous-tâche RÉDUIT PROGRESSIVEMENT le champ de recherche
                           - Minimiser le périmètre de la tâche suivante
                           - Créer une LOGIQUE DÉICTIQUE (qui se resserre)

                        2. CRITÈRES D'ORDONNANCEMENT :
                           - Commencer par les tâches qui ÉLARGISSENT le champ
                           - Progresser vers des tâches de plus en plus SPÉCIFIQUES
                           - Chaque étape LIMITE le périmètre de la suivante

                        3. STRATÉGIE DE RÉDUCTION :
                           - Tâche 1 : Vue large, exploration générale
                           - Tâche 2 : Filtrage, réduction du champ
                           - Tâche N : Précision maximale, résultat final

                        4. CONTRAINTES D'EXÉCUTION :
                           - Chaque sous-tâche UTILISE les résultats de la précédente
                           - Réduire EXPONENTIELLEMENT le périmètre de recherche
                           - Garantir une progression logique et efficace

                        FORMAT IMPÉRATIF :
                        [
                            {
                                "task": "Description précise",
                                "agent": "Agent optimal",
                                "priority": "haute/moyenne/basse",
                                "contribution": "Rôle dans l'objectif global",
                                "input_constraints": ["résultat tâche précédente"],
                                "output_constraints": ["résultat à produire"],
                                "reduction_ratio": "Pourcentage de réduction du périmètre"
                            }
                        ]
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"""
                        Requête originale : {request}
                        
                        OBJECTIF GLOBAL :
                        - Description : {global_objective_data.get('global_objective', 'Non défini')}
                        - Type : {global_objective_data.get('objective_type', 'Non spécifié')}
                        - Dimensions clés : {', '.join(global_objective_data.get('key_dimensions', []))}
                        - Contraintes initiales : {', '.join(global_objective_data.get('initial_constraints', []))}
                        
                        CONSIGNE CRUCIALE : 
                        - Décompose en sous-tâches séquentielles
                        - CHAQUE sous-tâche UTILISE les résultats des tâches précédentes
                        - Réduire PROGRESSIVEMENT le périmètre de recherche
                        - Chaque résultat LIMITE le champ de la tâche suivante
                        """
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.4
            )
            
            result = json.loads(response.choices[0].message.content)
            subtasks = result.get('subtasks', [])
            
            return subtasks
        
        except Exception as e:
            logger.error(f"Erreur de génération de sous-tâches : {e}")
            return [{
                'task': "Analyse stratégique globale",
                'agent': 'Multi-Purpose Agent',
                'priority': 'haute',
                'contribution': f"Décomposition et résolution de la requête complexe : {request}",
                'input_constraints': [],
                'output_constraints': ["Fournir une analyse complète"]
            }]

    def _get_dict_value(self, obj: Any, key: str, default: Any = None) -> Any:
        """
        Récupérer une valeur d'un dictionnaire de manière sécurisée
        """
        try:
            if hasattr(obj, 'dict'):
                # Pour les objets Pydantic
                return obj.dict().get(key, default)
            elif hasattr(obj, 'get'):
                # Pour les dictionnaires
                return obj.get(key, default)
            elif hasattr(obj, key):
                # Pour les objets avec attributs
                return getattr(obj, key)
            else:
                return default
        except Exception:
            return default

    def _extract_content(self, result: Any) -> str:
        """
        Extraire le contenu d'un résultat de différents types avec une gestion robuste
        
        Args:
            result (Any): Le résultat à extraire
        
        Returns:
            str: Le contenu extrait
        """
        # Gestion explicite des RunResponse
        if hasattr(result, 'content'):
            # Vérifier le type de contenu
            if isinstance(result.content, str):
                return result.content
            elif isinstance(result.content, dict):
                return json.dumps(result.content)
            elif result.content is None:
                return ""
            else:
                return str(result.content)
        
        # Autres types de résultats
        if result is None:
            return ""
        
        # Gestion des dictionnaires et listes
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        
        # Conversion en chaîne par défaut
        return str(result)

    async def decompose_task(self, user_request: str) -> TaskLedger:
        """
        Décomposer la requête utilisateur en sous-tâches avec function calling
        
        Args:
            user_request (str): La requête utilisateur originale
        
        Returns:
            TaskLedger: Le registre de tâches mis à jour
        """
        # 1. Vérification préliminaire de décomposition
        needs_decomposition = self.should_decompose_task(user_request)
        
        # Ajouter un fait sur la décision de décomposition
        self.task_ledger.add_fact(
            f"Décomposition nécessaire : {needs_decomposition}", 
            fact_type="derived"
        )
        
        # 2. Si pas besoin de décomposition, traitement simple
        if not needs_decomposition:
            # Ajouter directement la tâche au registre
            self.task_ledger.add_task(user_request)
            logger.info(f"""
Task Ledger généré :
    - Requête utilisateur : {user_request}
    - Nombre de sous-tâches : 1
""")
            return self.task_ledger
        
        # 3. Logique existante de décomposition avec function calling
        try:
            # Appel au LLM pour décomposer la tâche
            response = self.client.chat.completions.create(
                model=self.llm_config["model"],
                messages=[
                    {"role": "system", "content": "Décompose la tâche en sous-tâches précises et ordonnées"},
                    {"role": "user", "content": user_request}
                ],
                functions=self._get_task_decomposition_functions(),
                function_call={"name": "decompose_task"}
            )
            
            # 4. Parsing de la réponse
            decomposition = self._parse_json_response(response)
            
            # 5. Ajouter les sous-tâches au registre
            for subtask in decomposition:
                self.task_ledger.add_task(subtask['task'])
                
                # Ajouter un fait sur chaque sous-tâche
                self.task_ledger.add_fact(
                    f"Sous-tâche identifiée : {subtask['task']}", 
                    fact_type="derived"
                )
            
            logger.info(f"""
                Task Ledger généré :
                - Requête utilisateur : {user_request}
                - Nombre de sous-tâches : {len(decomposition)}
                - Détail des sous-tâches :
                {json.dumps(decomposition, indent=2)}
                """)
            
            return self.task_ledger
        
        except Exception as e:
            # Gestion des erreurs
            logger.error(f"Erreur lors de la décomposition : {e}")
            
            # Repli : ajouter la tâche originale
            self.task_ledger.add_task(user_request)
            self.task_ledger.add_fact(
                f"Échec de décomposition, tâche originale conservée", 
                fact_type="guesses"
            )
        
        return self.task_ledger

    async def execute_task(self, task_ledger: TaskLedger) -> Dict[str, Any]:
        """
        Exécuter les sous-tâches de manière unifiée
        
        Args:
            task_ledger (TaskLedger): Le registre de tâches à exécuter
        
        Returns:
            Dict[str, Any]: Résultats de l'exécution des tâches
        """
        task_results = {}
        
        try:
            logger.info("📋 Début de l'exécution des sous-tâches")
            logger.info(f"🔢 Nombre total de sous-tâches : {len(task_ledger.current_plan)}")
            
            for task_index, current_task in enumerate(task_ledger.current_plan[:], 1):
                try:
                    # Sélection dynamique de l'agent
                    selected_agent = self._select_best_agent(current_task)
                    
                    # Exécution de la tâche
                    logger.info(f"🚀 Exécution de la sous-tâche {task_index}/{len(task_ledger.current_plan)}")
                    logger.info(f"📝 Tâche : {current_task}")
                    logger.info(f"🤖 Agent sélectionné : {selected_agent.name}")
                    
                    result = await selected_agent.arun(current_task)
                    
                    # Extraction du contenu
                    result_content = self._extract_content(result)
                    
                    # Stocker le résultat
                    task_results[current_task] = {
                        "agent": selected_agent.name,
                        "result": result_content
                    }
                    
                    logger.info(f"✅ Sous-tâche {task_index} terminée")
                    logger.info(f"📊 Résultat : {result_content[:200]}...")
                
                except Exception as task_error:
                    logger.error(f"❌ Erreur lors de l'exécution de la sous-tâche {task_index} : {task_error}")
                    logger.error(traceback.format_exc())
                    
                    task_results[current_task] = {
                        "error": str(task_error),
                        "traceback": traceback.format_exc()
                    }
            
            logger.info("🏁 Exécution de toutes les sous-tâches terminée")
            
            return task_results
        
        except Exception as global_error:
            logger.error(f"❌ Erreur globale lors de l'exécution des tâches : {global_error}")
            logger.error(traceback.format_exc())
            
            return {
                "error": str(global_error),
                "task_results": task_results
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
            # Décomposition de la tâche
            task_ledger = await self.decompose_task(user_request)
            
            # Exécution des sous-tâches
            task_results = await self.execute_task(task_ledger)
            
            # Synthèse des résultats
            synthesis_result = await self._synthesize_results(task_results)
            
            return {
                "task_ledger": task_ledger.to_json(),
                "task_results": task_results,
                "final_result": synthesis_result
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de la requête : {e}")
            logger.error(traceback.format_exc())
            
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    async def _synthesize_results(self, task_results: Dict[str, Any]) -> str:
        """
        Synthétiser les résultats des sous-tâches
        
        Args:
            task_results (Dict[str, Any]): Résultats des sous-tâches
        
        Returns:
            str: Synthèse des résultats
        """
        try:
            synthesis_prompt = TASK_CONTEXT_PROMPT.format(
                task=self.task_ledger.original_request,
                context=json.dumps(task_results)
            )
            
            return self.agent.run(synthesis_prompt)
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la synthèse des résultats : {e}")
            return "Impossible de synthétiser les résultats."

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
        return result
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête : {e}")
        logger.error(traceback.format_exc())
        return {
            "error": f"Erreur lors du traitement de la requête : {e}",
            "query": user_request,
            "result": None
        }

def get_orchestrator_agent(
    model_id: str = "gpt-4o-mini"
) -> Agent:
    """
    Créer et configurer l'agent orchestrateur principal
    
    Args:
        model_id (str): Identifiant du modèle OpenAI à utiliser
    
    Returns:
        Agent: Agent orchestrateur configuré
    """
    web_agent = get_web_searcher(
        model_id=agent_settings.gpt_4,
        debug_mode=False
    )
    api_knowledge_agent = get_api_knowledge_agent(
        debug_mode=False,
        user_id=None,
        session_id=None
    )
    data_analysis_agent = get_data_analysis_agent(
        debug_mode=False
    )
    travel_planner = get_travel_planner(
        debug_mode=False
    )

    return Agent(
        name="Advanced Task Orchestrator",
        role="Décomposer et coordonner des tâches complexes de manière autonome",
        model=OpenAIChat(
            id=model_id,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=0.7  # Plus créatif pour la décomposition
        ),
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
        tools=[
            PythonTools(),
            DuckDuckGo()
        ],
        team=[
            web_agent,
            api_knowledge_agent,
            data_analysis_agent,
            travel_planner
        ]
    )

# Exemple d'utilisation
if __name__ == "__main__":
    import asyncio
    async def main():
        test_request = "Trouve des informations sur l'intelligence artificielle et résume-les"
        result = await process_user_request(test_request, debug_mode=True)
        print(result)
    asyncio.run(main())