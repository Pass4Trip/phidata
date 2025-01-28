import asyncio
import logging
from typing import Dict, Any, Optional, Callable

import pytest
from phi.agent import Agent

from agents.user_proxy import get_user_proxy_agent

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Gestionnaires de tâches simulés
async def async_web_search_handler(task_info: Dict[str, Any]) -> str:
    """Simulation d'une recherche web asynchrone"""
    logger.info(f"🌐 Démarrage recherche web : {task_info['task']}")
    await asyncio.sleep(2)  # Simulation temps de recherche
    return f"Résultats pour {task_info['task']}"

async def async_data_analysis_handler(task_info: Dict[str, Any]) -> str:
    """Simulation d'une analyse de données asynchrone"""
    logger.info(f"📊 Démarrage analyse : {task_info['task']}")
    await asyncio.sleep(3)  # Simulation temps d'analyse
    return f"Rapport pour {task_info['task']}"

async def async_document_processing_handler(task_info: Dict[str, Any]) -> str:
    """Simulation de traitement de document"""
    logger.info(f"📄 Démarrage traitement document : {task_info['task']}")
    await asyncio.sleep(4)  # Simulation temps de traitement
    return f"Document traité : {task_info['task']}"

class TestUserProxyAgent:
    @pytest.fixture
    def user_proxy_agent(self):
        """Fixture pour créer un agent UserProxy pour chaque test"""
        return get_user_proxy_agent(
            user_id="test_user", 
            session_id="test_session", 
            debug_mode=True
        )

    @pytest.mark.asyncio
    async def test_multiple_async_tasks(self, user_proxy_agent):
        """
        Test de traitement de plusieurs tâches asynchrones
        Scénario : Plusieurs requêtes de types différents
        """
        # Récupérer les outils spécifiques
        add_task_tool = next(tool for tool in user_proxy_agent.tools if tool.__name__ == 'add_user_request_to_ledger')
        process_tasks_tool = next(tool for tool in user_proxy_agent.tools if tool.__name__ == 'process_user_tasks')

        # Définir un user_id de test
        test_user_id = "test_user_123"

        # Ajout de tâches avec différents gestionnaires
        add_task_tool(
            "Rechercher des informations sur l'IA", 
            user_id=test_user_id,
            handler=async_web_search_handler
        )
        add_task_tool(
            "Analyser les tendances technologiques", 
            user_id=test_user_id,
            handler=async_data_analysis_handler
        )
        add_task_tool(
            "Traiter un rapport technique", 
            user_id=test_user_id,
            handler=async_document_processing_handler
        )
        add_task_tool(
            "Recherche complémentaire sur robotique", 
            user_id=test_user_id,
            handler=async_web_search_handler
        )

        # Traitement des tâches
        results = await process_tasks_tool(max_concurrent_tasks=3, user_id=test_user_id)
        
        # Vérifications
        assert len(results) == 4, "Toutes les tâches doivent être traitées"
        logger.info("✅ Test de traitement asynchrone réussi")

    @pytest.mark.asyncio
    async def test_task_ledger_tracking(self, user_proxy_agent):
        """
        Test du suivi précis des tâches dans le TaskLedger
        Scénario : Vérification du statut et de l'historique des tâches
        """
        # Récupérer les outils spécifiques
        add_task_tool = next(tool for tool in user_proxy_agent.tools if tool.__name__ == 'add_user_request_to_ledger')
        get_next_task_tool = next(tool for tool in user_proxy_agent.tools if tool.__name__ == 'get_next_pending_task')
        process_tasks_tool = next(tool for tool in user_proxy_agent.tools if tool.__name__ == 'process_user_tasks')

        # Définir un user_id de test
        test_user_id = "test_user_456"

        # Ajout de tâches
        add_task_tool(
            "Première tâche de test", 
            user_id=test_user_id,
            handler=async_web_search_handler
        )
        add_task_tool(
            "Deuxième tâche de test", 
            user_id=test_user_id,
            handler=async_data_analysis_handler
        )

        # Récupération de la prochaine tâche en attente
        next_task = get_next_task_tool(user_id=test_user_id)
        assert next_task is not None, "Une tâche en attente doit exister"

        # Traitement des tâches
        await process_tasks_tool(user_id=test_user_id)

        # Vérification du TaskLedger
        task_ledger = user_proxy_agent.task_ledger
        assert task_ledger.is_all_tasks_completed(), "Toutes les tâches doivent être terminées"
        
        # Vérifier que seules les tâches de l'utilisateur sont comptées
        user_tasks = [
            task for task in task_ledger.task_history 
            if task.get('context', {}).get('user_id') == test_user_id
        ]
        assert len(user_tasks) == 2, "Deux tâches doivent avoir été traitées pour cet utilisateur"
        
        logger.info("✅ Test de suivi des tâches réussi")

    def test_agent_initialization(self, user_proxy_agent):
        """
        Test de l'initialisation de l'agent UserProxy
        Scénario : Vérification des attributs et outils
        """
        # Vérification des outils
        tool_names = [tool.__name__ for tool in user_proxy_agent.tools]
        assert 'add_user_request_to_ledger' in tool_names
        assert 'get_next_pending_task' in tool_names
        assert 'process_user_tasks' in tool_names

        # Vérification du TaskLedger
        assert hasattr(user_proxy_agent, 'task_ledger')
        assert user_proxy_agent.task_ledger is not None

        logger.info("✅ Test d'initialisation de l'agent réussi")

if __name__ == "__main__":
    pytest.main([__file__])
