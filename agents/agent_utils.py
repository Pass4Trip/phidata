import logging
import functools
import time

def log_agent_method(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        agent_name = args[0].__class__.__name__ if len(args) > 0 else "Unknown Agent"
        task = kwargs.get('task', args[1] if len(args) > 1 else "No task")
        
        logging.info(f"🤖 Agent {agent_name} - Début d'exécution")
        logging.info(f"📋 Tâche : {task}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            logging.info(f"✅ Agent {agent_name} - Exécution terminée")
            logging.info(f"⏱️ Temps d'exécution : {execution_time:.2f} secondes")
            
            return result
        
        except Exception as e:
            logging.error(f"❌ Agent {agent_name} - Erreur d'exécution")
            logging.error(f"🔴 Détails : {str(e)}")
            raise
    
    return wrapper
