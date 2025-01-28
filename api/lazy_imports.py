import importlib
import logging
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

def lazy_import(module_path: str, item_name: Optional[str] = None) -> Callable[..., Any]:
    """
    Importe un module ou un élément de module de manière paresseuse.
    
    Args:
        module_path (str): Chemin complet du module
        item_name (str, optional): Nom spécifique à importer du module
    
    Returns:
        Callable: Fonction qui importe et retourne le module/élément à la demande
    """
    @wraps(lazy_import)
    def wrapper(*args, **kwargs):
        try:
            # Importer le module
            module = importlib.import_module(module_path)
            
            # Si un item spécifique est demandé
            if item_name:
                item = getattr(module, item_name)
                
                # Si c'est un callable, l'appeler avec les arguments
                if callable(item):
                    return item(*args, **kwargs)
                return item
            
            return module
        
        except (ImportError, AttributeError) as e:
            logger.error(f"Erreur d'import pour {module_path}.{item_name or ''}: {e}")
            raise
    
    return wrapper

# Exemples de lazy imports
get_web_searcher = lazy_import('agents.web', 'get_web_searcher')
get_api_knowledge_agent = lazy_import('agents.api_knowledge', 'get_api_knowledge_agent')
get_user_proxy_agent = lazy_import('agents.user_proxy', 'get_user_proxy_agent')
# Ajoutez d'autres imports paresseux selon vos besoins
