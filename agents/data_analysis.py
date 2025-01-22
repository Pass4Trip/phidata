from typing import Optional
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.python import PythonTools
from agents.settings import agent_settings
import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from db.url import get_db_url
from phi.storage.agent.postgres import PgAgentStorage
from phi.agent import AgentMemory
from phi.memory.db.postgres import PgMemoryDb
from utils.colored_logging import get_colored_logger

# Charger les variables d'environnement
load_dotenv()

# Configuration du logger
logger = get_colored_logger('agents.data_analysis', 'DataAnalysisAgent', level=logging.DEBUG)

db_url = get_db_url()

data_analysis_storage = PgAgentStorage(table_name="data_analysis_sessions", db_url=db_url)

def get_data_analysis_agent(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    # Vérifier si la clé API est définie
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("La clé API OpenAI n'est pas définie. Veuillez définir OPENAI_API_KEY dans votre fichier .env.")

    # Configurer le logging en mode DEBUG
    logger.setLevel(logging.INFO)

    # Récupérer l'URL de base de données (optionnel)
    if db_url:
        logger.debug(f"URL de base de données configurée : {db_url}")



    def load_dataframe(file_path):
        """
        Charger un fichier de données
        """
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                return pd.read_excel(file_path)
            else:
                return f"Format de fichier non supporté : {file_path}"
        except Exception as e:
            return f"Erreur de chargement : {str(e)}"

    def analyze_dataframe(df):
        """
        Analyser un dataframe et générer des insights
        """
        if not isinstance(df, pd.DataFrame):
            return "Données invalides. Veuillez fournir un DataFrame pandas."
        
        analysis_results = {
            "shape": df.shape,
            "columns": list(df.columns),
            "statistical_summary": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        return analysis_results

    data_analysis_agent = Agent(
        name="Data Analysis Agent",
        agent_id="data-analysis-agent",
        session_id=session_id,
        user_id=user_id,
        model=OpenAIChat(
            id=model_id or agent_settings.gpt_4,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        role="Tu es un agent spécialisé dans l'analyse de données. Tu dois fournir des insights détaillés et des analyses statistiques.",
        instructions=[
            "Effectuer des analyses de données approfondies",
            "Utiliser pandas et numpy pour le traitement des données",
            "Générer des visualisations et des insights statistiques",
            "Expliquer les résultats de manière claire et compréhensible",
            "",
            "Étapes de résolution :",
            "1. Charger et valider les données",
            "2. Effectuer une analyse statistique descriptive",
            "3. Identifier les tendances et les insights clés",
            "",
            "Règles importantes :",
            " - Toujours vérifier la qualité et la cohérence des données",
            " - Expliquer les méthodes d'analyse utilisées",
            " - Fournir des recommandations basées sur les données",
            " - Être transparent sur les limites de l'analyse",
        ],
        tools=[
            PythonTools(),
            load_dataframe,
            analyze_dataframe
        ],
        add_datetime_to_instructions=True,
        markdown=True,
        show_tool_calls=True,
        monitoring=True,
        debug_mode=debug_mode,
        storage=data_analysis_storage,
        memory=AgentMemory(
            db=PgMemoryDb(table_name="data_analysis_memory", db_url=db_url),
            create_user_memories=True,
            create_session_summary=True
        ),
        stream=False  # Désactiver le streaming
    )

    def perform_data_analysis(query, data_source=None):
        """
        Méthode utilitaire pour l'analyse de données
        """
        if data_source:
            df = load_dataframe(data_source)
            analysis = analyze_dataframe(df)
            return data_analysis_agent.print_response(
                f"{query}\n\nRésultats d'analyse : {analysis}", 
                stream=True
            )
        else:
            return data_analysis_agent.print_response(query, stream=True)

    return data_analysis_agent

# Exemple d'utilisation
if __name__ == "__main__":
    test_query = "Analyser les tendances dans ce jeu de données"
    data_analysis_agent = get_data_analysis_agent()
    data_analysis_agent.perform_data_analysis(test_query, "example_data.csv")
