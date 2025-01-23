from typing import Optional, Any, Dict, List
import os
import logging
from dotenv import load_dotenv
import json

from phi.agent import Agent
from phi.llm.openai.chat import OpenAIChat
from phi.tools.python import PythonTools
from phi.tools.duckduckgo import DuckDuckGo

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logger = logging.getLogger(__name__)

class TravelPlannerAgent:
    def __init__(
        self, 
        model_id: str = "gpt-4o-mini", 
        debug_mode: bool = False
    ):
        """
        Initialise l'agent de planification de voyage.
        
        Args:
            model_id (str): Identifiant du modèle OpenAI à utiliser
            debug_mode (bool): Mode de débogage
        """
        # Configuration du modèle LLM
        llm_config = {
            "model": model_id,
            "temperature": 0.2
        }

        # Créer l'agent de voyage
        self.agent = Agent(
            llm=OpenAIChat(**llm_config),
            tools=[
                PythonTools(),
                DuckDuckGo()
            ],
            instructions=[
                "Tu es un agent spécialisé dans la planification de voyages.",
                "Ton objectif est de fournir des informations détaillées et pratiques.",
                "Étapes de travail :",
                "1. Analyser la requête de voyage",
                "2. Rechercher des informations pertinentes",
                "3. Structurer une réponse claire et utile",
                "4. Proposer des recommandations concrètes"
            ],
            debug_mode=debug_mode
        )

    async def plan_itinerary(
        self,
        destination: str,
        start_date: str,
        end_date: str,
        budget: float,
        preferences: List[str]
    ) -> Dict[str, Any]:
        """
        Planifier un itinéraire de voyage
        
        Args:
            destination (str): Destination du voyage
            start_date (str): Date de début du voyage
            end_date (str): Date de fin du voyage
            budget (float): Budget total du voyage
            preferences (List[str]): Liste des préférences de voyage
        
        Returns:
            Dict[str, Any]: Détails de l'itinéraire
        """
        try:
            # Utiliser l'agent pour planifier l'itinéraire
            result = await self.agent.run(
                f"Planifier un voyage à {destination} du {start_date} au {end_date} "
                f"avec un budget de {budget}€. "
                f"Préférences : {', '.join(preferences)}"
            )
            
            return {
                "destination": destination,
                "start_date": start_date,
                "end_date": end_date,
                "budget": budget,
                "preferences": preferences,
                "itinerary": result
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la planification de l'itinéraire : {e}")
            return {
                "error": str(e)
            }

    async def get_destination_recommendations(
        self,
        travel_type: str,
        preferences: List[str],
        budget: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtenir des recommandations de destinations
        
        Args:
            travel_type (str): Type de voyage (aventure, détente, culturel, etc.)
            preferences (List[str]): Liste des préférences de voyage
            budget (Optional[float]): Budget optionnel
        
        Returns:
            List[Dict[str, Any]]: Liste des destinations recommandées
        """
        try:
            # Utiliser l'agent pour obtenir des recommandations
            budget_info = f"avec un budget de {budget}€" if budget else ""
            result = await self.agent.run(
                f"Recommander des destinations de voyage {travel_type} "
                f"{budget_info}. Préférences : {', '.join(preferences)}"
            )
            
            # Convertir le résultat en liste de destinations
            destinations = [
                {"name": dest.strip()} for dest in result.split("\n") if dest.strip()
            ]
            
            return destinations
        
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de destinations : {e}")
            return []

def get_travel_planner(
    model_id: str = "gpt-4o-mini",
    debug_mode: bool = False,
    **kwargs
) -> Agent:
    """
    Crée et configure un agent de planification de voyage.
    
    Args:
        model_id (str): Identifiant du modèle OpenAI
        debug_mode (bool): Mode de débogage
    
    Returns:
        Agent: Agent de planification de voyage configuré
    """
    travel_planner = TravelPlannerAgent(
        model_id=model_id, 
        debug_mode=debug_mode
    )
    
    # Ajouter un nom explicite à l'agent
    travel_planner.agent.name = "Travel Planner Agent"
    
    return travel_planner.agent
