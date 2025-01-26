import os
import logging
import pytest
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Importer l'agent web
from agents.web import get_web_searcher

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_web_search_agent():
    """
    Test explicite de l'agent de recherche web
    """
    # Vérifier que la clé API OpenAI est disponible
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None, "La clé API OpenAI est manquante"

    # Créer l'agent web
    web_agent = get_web_searcher(
        model_id="gpt-4o-mini", 
        debug_mode=True
    )

    # Liste de requêtes de test
    test_queries = [
        "Quelle est la capitale de la France ?",
        "Météo à Paris cette semaine",
        "Dernières nouvelles technologiques"
    ]

    # Tester chaque requête
    for query in test_queries:
        logger.info(f"🔍 Test de recherche : {query}")
        
        try:
            # Exécuter la recherche
            result = web_agent.run(query)
            
            # Vérifications
            assert result is not None, f"Aucun résultat pour la requête : {query}"
            assert len(str(result)) > 0, f"Résultat vide pour la requête : {query}"
            
            logger.info(f"✅ Recherche réussie pour : {query}")
            logger.debug(f"📋 Résultat : {result}")
        
        except Exception as e:
            logger.error(f"❌ Échec de la recherche pour {query}: {e}")
            raise

    logger.info("🎉 Tous les tests de l'agent web sont passés avec succès !")

# Permet de lancer le test directement
if __name__ == "__main__":
    pytest.main([__file__])
