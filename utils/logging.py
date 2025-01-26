import logging
import os
from typing import Optional

def configure_logging(log_level: Optional[str] = None):
    """
    Configure le logging avec un niveau personnalisable.
    
    Args:
        log_level (Optional[str]): Niveau de log. 
        Valeurs possibles : 'INFO', 'DEBUG', None (par défaut)
    """
    # Récupérer le niveau de log depuis une variable d'environnement si non spécifié
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
    
    # Mapper les niveaux de log
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # Sélectionner le niveau de log
    selected_level = log_levels.get(log_level, logging.DEBUG)
    
    # Configuration du logging
    logging.basicConfig(
        level=selected_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Ajouter un handler de console
        ]
    )
    
    # Configurer les loggers spécifiques pour réduire le bruit de débogage
    loggers_to_quiet = [
        'httpx',  # Réduire les logs des requêtes HTTP
        'openai',  # Réduire les logs OpenAI
        'phi',     # Réduire les logs Phidata
    ]
    
    for logger_name in loggers_to_quiet:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
    
    # Log le niveau de log configuré
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configuré au niveau : {log_level}")
    
    return selected_level
