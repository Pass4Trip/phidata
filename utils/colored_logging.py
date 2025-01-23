import logging
import colorama
from colorama import Fore, Style

# Initialiser colorama
colorama.init(autoreset=True)

# Définition des couleurs par agent
AGENT_COLORS = {
    'MainRouterAgent': Fore.CYAN,
    'WebAgent': Fore.GREEN,
    'APIKnowledgeAgent': Fore.MAGENTA,
    'DataAnalysisAgent': Fore.YELLOW,
    'DefaultAgent': Fore.WHITE,
    'Orchestrator': Fore.CYAN
}

class ColoredFormatter(logging.Formatter):
    """
    Formateur de log personnalisé avec coloration par agent
    """
    def format(self, record):
        # Récupérer la couleur de l'agent
        agent_name = getattr(record, 'agent_name', 'DefaultAgent')
        color = AGENT_COLORS.get(agent_name, AGENT_COLORS['DefaultAgent'])
        
        # Formatage du message
        log_message = super().format(record)
        
        # Coloration du message
        colored_message = f"{color}{log_message}{Style.RESET_ALL}"
        
        return colored_message

def get_colored_logger(name, agent_name=None, level=logging.INFO):
    """
    Créer un logger coloré pour un agent spécifique
    
    Args:
        name (str): Nom du logger
        agent_name (str, optional): Nom de l'agent pour la coloration
        level (int, optional): Niveau de logging
    
    Returns:
        logging.Logger: Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Créer un handler de console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Créer un formateur coloré
    formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Ajouter le handler
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    # Ajouter un attribut pour le nom de l'agent
    if agent_name:
        logger.agent_name = agent_name
    
    return logger

# Exemples d'utilisation
def demo_logging():
    """
    Démonstration de l'utilisation du logging coloré
    """
    main_logger = get_colored_logger('agents.orchestrator', 'Orchestrator')
    web_logger = get_colored_logger('agents.web', 'WebAgent')
    api_logger = get_colored_logger('agents.api_knowledge', 'APIKnowledgeAgent')
    
    main_logger.info("Message du routeur principal")
    web_logger.warning("Avertissement de l'agent web")
    api_logger.error("Erreur de l'agent de connaissances API")

if __name__ == '__main__':
    demo_logging()
