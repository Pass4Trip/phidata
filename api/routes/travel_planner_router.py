from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel

from agents.travel_planner import get_travel_planner

travel_planner_router = APIRouter(prefix="/travel", tags=["Travel Planning"])

class DestinationCriteria(BaseModel):
    budget_range: tuple[float, float]
    duration_range: tuple[int, int]
    preferred_climate: str
    activities: List[str]
    travel_style: str
    season: str

class TravelPlan(BaseModel):
    destination: str
    duration: int
    budget: float
    preferences: List[str]

@travel_planner_router.post("/search-destinations")
async def search_destinations(
    criteria: DestinationCriteria
) -> Dict[str, Any]:
    """
    Rechercher des destinations selon les critères spécifiés
    """
    try:
        agent = get_travel_planner()
        return await agent.search_destinations(criteria.dict())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la recherche de destinations : {str(e)}"
        )

@travel_planner_router.post("/plan-itinerary")
async def plan_itinerary(
    plan: TravelPlan
) -> Dict[str, Any]:
    """
    Planifier un itinéraire détaillé pour un voyage
    """
    try:
        agent = get_travel_planner()
        return await agent.plan_itinerary(
            destination=plan.destination,
            duration=plan.duration,
            budget=plan.budget,
            preferences=plan.preferences
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la planification de l'itinéraire : {str(e)}"
        )
