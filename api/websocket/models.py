from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class WebSocketMessage(BaseModel):
    """
    Modèle standardisé pour les messages WebSocket
    """
    type: str = Field(..., description="Type de message (request, response, error)")
    user_id: str = Field(..., description="Identifiant de l'utilisateur")
    session_id: Optional[str] = Field(None, description="ID de session optionnel")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    content: Dict[str, Any] = Field(..., description="Contenu du message")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class WebSocketResponse(BaseModel):
    """
    Modèle de réponse standardisé
    """
    status: str = Field(..., description="Statut de la réponse (success, error)")
    message: Optional[str] = Field(None, description="Message descriptif")
    data: Optional[Dict[str, Any]] = Field(None, description="Données de réponse")
