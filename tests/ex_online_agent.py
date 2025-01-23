import sys
import os

# Ajouter le chemin du projet au PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Charger explicitement les variables d'environnement
from dotenv import load_dotenv
load_dotenv()

from llm_axe.models import OpenAIChat
from llm_axe.agents import OnlineAgent

# Example showing how to use an online agent
# The online agent will use the internet to try and best answer the user prompt
prompt = "Donne moi une news d'aujourd'hui 23/01/2025"

# Récupérer la clé API de manière sécurisée
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Erreur : Clé API OpenAI manquante. Veuillez la définir dans le fichier .env")
    sys.exit(1)

try:
    llm = OpenAIChat(api_key=api_key)
    searcher = OnlineAgent(llm)
    resp = searcher.search(prompt)
    print(resp)
except Exception as e:
    print(f"Erreur lors de la recherche : {e}")
    sys.exit(1)
