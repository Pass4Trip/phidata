from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import traceback
import re

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.python import PythonTools
from phi.tools.duckduckgo import DuckDuckGo

from agents.settings import agent_settings
from agents.web import get_web_searcher
from agents.api_knowledge import get_api_knowledge_agent
from agents.data_analysis import get_data_analysis_agent
from agents.orchestrator_prompts import (
    get_task_decomposition_prompt,
    get_task_execution_prompt,
    get_task_synthesis_prompt
)

from utils.colored_logging import get_colored_logger

import asyncio

logger = get_colored_logger('agents.orchestrator', 'OrchestratorAgent')

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
            "updated_at": self.updated_at.isoformat()
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
        user_id: Optional[str] = None, 
        session_id: Optional[str] = None, 
        debug_mode: bool = False
    ):
        """
        Initialiser l'agent orchestrateur avec des agents spécialisés
        
        Args:
            user_id (Optional[str]): ID de l'utilisateur
            session_id (Optional[str]): ID de session
            debug_mode (bool): Mode de débogage
        """
        # Initialiser les agents spécialisés
        self.web_agent = get_web_searcher(
            model_id=agent_settings.gpt_4,
            user_id=user_id,
            session_id=session_id,
            debug_mode=debug_mode
        )
        self.api_knowledge_agent = get_api_knowledge_agent(
            user_id=user_id,
            session_id=session_id,
            debug_mode=debug_mode
        )
        self.data_analysis_agent = get_data_analysis_agent(
            user_id=user_id,
            session_id=session_id,
            debug_mode=debug_mode
        )

        # Agent orchestrateur principal
        self.agent = get_orchestrator_agent(
            user_id=user_id,
            session_id=session_id,
            debug_mode=debug_mode
        )
        
        # Liste de tous les agents disponibles
        self.agents = [
            self.web_agent, 
            self.api_knowledge_agent, 
            self.data_analysis_agent, 
            self.agent
        ]

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

    async def _generate_dynamic_agent_selection(self, task: str, task_ledger: TaskLedger) -> Agent:
        """
        Sélectionner dynamiquement l'agent le plus approprié pour une tâche
        
        Args:
            task (str): La description de la tâche
            task_ledger (TaskLedger): Le registre de tâches pour le contexte
        
        Returns:
            Agent: L'agent sélectionné pour la tâche
        """
        # Scoring des agents basé sur plusieurs critères
        def score_agent(agent):
            # Performance historique
            agent_performance = self._get_dict_value(task_ledger.facts, 'agent_performance', {})
            performance = self._get_dict_value(agent_performance, agent.name, {
                'success_rate': 0.5,
                'tasks_completed': 0,
                'recent_performance': []
            })
            
            # Scoring par mots-clés
            keywords = {
                "Web Search Agent": ["rechercher", "destinations", "web", "internet"],
                "API Knowledge Agent": ["planning", "détails", "connaissances", "api"],
                "Data Analysis Agent": ["budget", "analyse", "calcul", "prévisionnel"]
            }
            
            keyword_score = sum(
                1 for keyword in keywords.get(agent.name, []) 
                if keyword in task.lower()
            )
            
            # Calcul du score
            recent_performance = self._get_dict_value(performance, 'recent_performance', [])
            recent_performance_score = len([p for p in recent_performance if p == 'success']) / (len(recent_performance) or 1)
            
            return (
                self._get_dict_value(performance, 'success_rate', 0.5) * 0.4 +  # Taux de succès historique
                recent_performance_score * 0.3 +                                 # Performance récente
                (keyword_score * 0.2) +                                         # Correspondance des mots-clés
                (self._get_dict_value(performance, 'tasks_completed', 0) * 0.1) # Expérience
            )
        
        try:
            # Sélectionner l'agent avec le meilleur score
            best_agent = max(self.agents, key=score_agent)
            logger.info(f"🎯 Agent sélectionné pour la tâche : {best_agent.name}")
            return best_agent
            
        except Exception as e:
            logger.warning(f"❗ Erreur lors de la sélection dynamique de l'agent : {e}")
            logger.info("🔄 Utilisation de l'agent par défaut")
            return self.agents[0]  # Agent par défaut

    def select_agent_for_task(self, current_task: str, task_ledger: TaskLedger) -> Agent:
        """
        Sélectionner dynamiquement l'agent le plus approprié pour une tâche
        Combine une approche par mots-clés et une sélection dynamique
        """
        # Définition des mots-clés pour chaque agent
        agent_keywords = {
            self.web_agent: ["recherche", "web", "internet", "actualité", "news"],
            self.api_knowledge_agent: ["api", "connaissance", "information", "synthèse"],
            self.data_analysis_agent: ["analyse", "données", "statistique", "calcul"]
        }
        
        # Première passe : sélection basée sur les mots-clés
        for agent, keywords in agent_keywords.items():
            if any(keyword in current_task.lower() for keyword in keywords):
                logger.info(f"🎯 Agent sélectionné par mots-clés pour la tâche '{current_task}': {agent.name}")
                return agent
        
        # Deuxième passe : utiliser un agent principal comme fallback
        logger.info(f"🔄 Aucun agent spécialisé trouvé. Utilisation de la sélection dynamique pour la tâche '{current_task}'")
        return self.agent

    async def execute_task(self, task_ledger: TaskLedger) -> Dict[str, Any]:
        """
        Exécuter les sous-tâches et suivre leur progression
        
        Args:
            task_ledger (TaskLedger): Le registre de tâches à exécuter
        
        Returns:
            Dict[str, Any]: Résultats de l'exécution des tâches
        """
        # Créer un registre de progression
        progress_ledger = ProgressLedger(task_ledger=task_ledger)
        
        try:
            # Copie de la liste des tâches pour éviter les modifications pendant l'itération
            for current_task in task_ledger.current_plan[:]:
                try:
                    # Réinitialiser le compteur de blocages
                    progress_ledger.reset_stall()
                    
                    # Sélection dynamique de l'agent
                    selected_agent = await self._generate_dynamic_agent_selection(current_task, task_ledger)
                    
                    # Exécution de la tâche
                    logger.info(f"🚀 Exécution de la tâche : {current_task}")
                    logger.info(f"🤖 Agent sélectionné : {selected_agent.name}")
                    
                    result = await selected_agent.arun(current_task)
                    
                    # Extraction robuste du contenu
                    result_content = self._extract_content(result)
                    
                    # Mise à jour du registre de progression
                    progress_ledger.complete_task(current_task, {
                        "result": result_content,
                        "agent": selected_agent.name
                    })
                    
                    # Vérifier la progression
                    if not progress_ledger.is_making_progress():
                        progress_ledger.increment_stall()
                    
                    # Vérifier si bloqué
                    if progress_ledger.is_stalled():
                        logger.warning("❗ Progression bloquée. Tentative de récupération.")
                        break
                
                except Exception as task_error:
                    # Gestion granulaire des erreurs par tâche
                    logger.error(f"❌ Erreur lors de l'exécution de la tâche '{current_task}': {task_error}")
                    logger.error(traceback.format_exc())
                    
                    progress_ledger.complete_task(current_task, {
                        "error": str(task_error),
                        "traceback": traceback.format_exc(),
                        "recovery_attempted": True
                    })
                    
                    # Incrémenter le compteur de blocages
                    progress_ledger.increment_stall()
                    
                    # Vérifier si bloqué
                    if progress_ledger.is_stalled():
                        logger.critical("🚨 Trop de blocages. Arrêt de l'exécution.")
                        break
            
            # Retourner le registre de progression
            return progress_ledger.to_json()
        
        except Exception as global_error:
            logger.critical(f"🚨 Erreur globale lors de l'exécution des tâches : {global_error}")
            logger.critical(traceback.format_exc())
            
            return {
                "error": str(global_error),
                "traceback": traceback.format_exc(),
                "status": "failed"
            }

    async def process_request(
        self, 
        user_request: str, 
        user_id: Optional[str] = None, 
        session_id: Optional[str] = None, 
        debug_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Traiter une requête utilisateur de bout en bout
        """
        logger.info(f"🌟 Début du traitement de la requête : '{user_request}'")
        
        # Décomposer la tâche
        task_ledger = self.decompose_task(user_request)
        
        # Exécuter les tâches
        results = await self.execute_task(task_ledger)
        
        # Préparer la réponse finale
        response = {
            "original_request": user_request,
            "results": results,
            "task_ledger": task_ledger.to_json(),
            "user_id": user_id
        }
        
        logger.info("🏁 Traitement de la requête terminé")
        
        return response

    def decompose_task(self, user_request: str) -> TaskLedger:
        """
        Décomposer la requête utilisateur en sous-tâches
        Ajoute la génération de faits initiaux
        """
        logger.info(f"🔍 Début de la décomposition de la tâche : '{user_request}'")
        task_ledger = TaskLedger(original_request=user_request)
        
        # Utiliser l'agent pour décomposer la tâche
        decomposition_prompt = get_task_decomposition_prompt(user_request)
        
        logger.info("📋 Génération du prompt de décomposition")
        decomposition_result = self.agent.run(decomposition_prompt)
        
        # Générer des faits initiaux
        facts_prompt = f"""
        Pour la requête : '{user_request}'
        
        Génère un ensemble de faits initiaux qui aideront à comprendre et à résoudre cette tâche.
        Inclus des informations contextuelles, des hypothèses de travail et des points clés à considérer.
        
        Format : Liste à puces concise et informative
        """
        task_ledger.facts = self.agent.run(facts_prompt)
        
        # Traiter le résultat de décomposition
        try:
            # Convertir le résultat en chaîne
            decomposition_str = self._extract_content(decomposition_result)
            
            # Extraire le JSON du bloc de code markdown
            import re
            match = re.search(r'```json\n(.*)\n```', decomposition_str, re.DOTALL)
            if match:
                decomposition_str = match.group(1)
            
            task_plan = json.loads(decomposition_str)
            
            # Extraire les sous-tâches, avec un fallback
            def extract_subtasks(plan):
                # Si le plan est un dictionnaire avec 'subtasks'
                if isinstance(plan, dict) and 'subtasks' in plan:
                    return plan['subtasks']
                # Si le plan est une liste
                elif isinstance(plan, list):
                    return plan
                # Si le plan est un dictionnaire sans 'subtasks'
                elif isinstance(plan, dict):
                    return list(plan.values())
                # Fallback
                else:
                    return [{"description": user_request, "suggested_agent": "API Knowledge Agent", "priority": "moyenne"}]
            
            subtasks = extract_subtasks(task_plan)
            
            # Normaliser les sous-tâches
            normalized_subtasks = []
            for subtask in subtasks:
                if isinstance(subtask, dict):
                    normalized_subtasks.append({
                        "description": subtask.get('description', user_request),
                        "suggested_agent": subtask.get('suggested_agent', 'API Knowledge Agent'),
                        "priority": subtask.get('priority', 'moyenne')
                    })
                elif isinstance(subtask, str):
                    normalized_subtasks.append({
                        "description": subtask,
                        "suggested_agent": "API Knowledge Agent",
                        "priority": "moyenne"
                    })
            
            # Si aucune sous-tâche n'a été trouvée, créer une tâche par défaut
            if not normalized_subtasks:
                normalized_subtasks = [{
                    "description": user_request,
                    "suggested_agent": "API Knowledge Agent",
                    "priority": "moyenne"
                }]
            
            logger.info("✅ Décomposition de la tâche réussie")
            for i, subtask in enumerate(normalized_subtasks, 1):
                logger.info(f"  🔸 Sous-tâche {i}: {subtask['description']} (Agent: {subtask['suggested_agent']}, Priorité: {subtask['priority']})")
            
            task_ledger.initial_plan = [subtask['description'] for subtask in normalized_subtasks]
            task_ledger.current_plan = task_ledger.initial_plan.copy()
        except (json.JSONDecodeError, TypeError) as e:
            # Fallback si le résultat n'est pas un JSON valide
            logger.warning(f"❌ Impossible de parser la décomposition de tâche : {e}")
            logger.warning(f"Résultat brut : {decomposition_result}")
            task_ledger.initial_plan = [user_request]
            task_ledger.current_plan = [user_request]
        
        return task_ledger

async def process_user_request(
    user_request: str, 
    user_id: Optional[str] = None, 
    session_id: Optional[str] = None, 
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Traiter une requête utilisateur de manière asynchrone avec l'orchestrateur
    
    Args:
        user_request (str): La requête de l'utilisateur
        user_id (Optional[str]): ID de l'utilisateur
        session_id (Optional[str]): ID de session
        debug_mode (bool): Mode de débogage
    
    Returns:
        Dict[str, Any]: Résultat du traitement de la requête
    """
    # Créer l'orchestrateur avec les agents
    orchestrator = OrchestratorAgent(
        user_id=user_id,
        session_id=session_id,
        debug_mode=debug_mode
    )
    
    # Traiter la requête de manière asynchrone
    return await orchestrator.process_request(
        user_request, 
        user_id, 
        session_id, 
        debug_mode=debug_mode
    )

def get_orchestrator_agent(
    user_id: Optional[str] = None, 
    session_id: Optional[str] = None, 
    debug_mode: bool = False
) -> Agent:
    """
    Créer et configurer l'agent orchestrateur principal
    
    Args:
        user_id (Optional[str]): ID de l'utilisateur
        session_id (Optional[str]): ID de session
        debug_mode (bool): Mode de débogage
    
    Returns:
        Agent: Agent orchestrateur configuré
    """
    web_agent = get_web_searcher(
        model_id=agent_settings.gpt_4,
        user_id=user_id,
        session_id=session_id,
        debug_mode=debug_mode
    )
    api_knowledge_agent = get_api_knowledge_agent(
        user_id=user_id,
        session_id=session_id,
        debug_mode=debug_mode
    )
    data_analysis_agent = get_data_analysis_agent(
        user_id=user_id,
        session_id=session_id,
        debug_mode=debug_mode
    )

    return Agent(
        name="Advanced Task Orchestrator",
        role="Décomposer et coordonner des tâches complexes de manière autonome",
        model=OpenAIChat(
            id=agent_settings.gpt_4,
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
            data_analysis_agent
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