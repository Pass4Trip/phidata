from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class WebSocketRequest(BaseModel):
    """
    Modèle de requête WebSocket standardisé
    """
    query: str = Field(..., description="Requête de l'utilisateur")
    user_id: str = Field(..., description="Identifiant de l'utilisateur")
    model_id: Optional[str] = Field(default="gpt-4o-mini", description="Modèle LLM à utiliser")
    session_id: Optional[str] = Field(None, description="ID de session")
    context: Optional[Dict[str, Any]] = Field(None, description="Contexte supplémentaire")

class WebSocketResponse(BaseModel):
    """
    Modèle de réponse WebSocket standardisé
    """
    status: str = Field(..., description="Statut de la réponse")
    message: Optional[str] = Field(None, description="Message descriptif")
    data: Optional[Dict[str, Any]] = Field(None, description="Données de réponse")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Horodatage de la réponse")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
