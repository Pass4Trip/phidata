import logging
from user_proxy import get_user_proxy_agent

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_task_ledger():
    """
    Test de la gestion des tâches dans le UserProxy Agent
    """
    # Initialiser l'agent UserProxy
    user_proxy_agent = get_user_proxy_agent(debug_mode=True)
    
    # Liste des tâches de test
    test_tasks = [
        "Rechercher des informations sur l'IA",
        "Planifier une réunion de projet",
        "Rédiger un rapport d'analyse"
    ]
    
    # Ajouter chaque tâche via l'outil de l'agent
    for task in test_tasks:
        logger.info(f"📝 Ajout de la tâche : {task}")
        user_proxy_agent.run(f"Ajouter la tâche : {task}")
    
    # Vérifier le contenu du TaskLedger
    task_ledger = user_proxy_agent.task_ledger
    
    logger.info("🔍 Détails du TaskLedger :")
    logger.info(f"Identifiant du registre : {task_ledger.ledger_id}")
    logger.info(f"Nombre total de tâches : {len(task_ledger.current_plan)}")
    logger.info(f"Liste des tâches : {task_ledger.current_plan}")
    
    # Vérifier la récupération de la prochaine tâche
    next_task = task_ledger.get_next_pending_task()
    logger.info(f"🏁 Prochaine tâche en attente : {next_task}")
    
    # Afficher le registre complet
    logger.info(repr(task_ledger))

if __name__ == "__main__":
    test_task_ledger()
