import os
import logging

logger = logging.getLogger(__name__)

def get_db_url() -> str | None:
    """
    Construit l'URL de connexion à la base de données à partir des variables d'environnement.
    
    Returns:
        Optional[str]: URL de connexion à la base de données, ou None si des variables sont manquantes
    """
    try:
        db_url = (
            f"postgresql://{os.getenv('DB_USER')}:"
            f"{os.getenv('DB_PASSWORD')}@"
            f"{os.getenv('DB_HOST')}:"
            f"{os.getenv('DB_PORT')}/"
            f"{os.getenv('DB_NAME')}"
        )
        
        # Vérifier que toutes les variables sont présentes
        if all([
            os.getenv('DB_USER'), 
            os.getenv('DB_PASSWORD'), 
            os.getenv('DB_HOST'), 
            os.getenv('DB_PORT'), 
            os.getenv('DB_NAME')
        ]):
            return db_url
        
        logger.warning("Une ou plusieurs variables de base de données sont manquantes.")
        return None
    
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'URL de base de données : {e}")
        return None
