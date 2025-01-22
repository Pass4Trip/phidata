from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from typing import Optional, Dict, Any
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_429_TOO_MANY_REQUESTS
from pydantic import BaseModel, Field
import logging
import json
import re
import os
import tempfile

from agents.data_analysis import get_data_analysis_agent

logger = logging.getLogger(__name__)
data_analysis_router = APIRouter()

def clean_response(result_str: str) -> str:
    """
    Nettoie la réponse pour la rendre plus lisible
    
    Args:
        result_str (str): Chaîne de résultat brute
    
    Returns:
        str: Chaîne de résultat nettoyée
    """
    # Supprimer les délimiteurs LaTeX
    result_str = re.sub(r'\\[()[\]]', '', result_str)
    
    # Supprimer les espaces en trop
    result_str = re.sub(r'\s+', ' ', result_str).strip()
    
    return result_str

class DataAnalysisResponse(BaseModel):
    """
    Modèle structuré pour la réponse d'analyse de données
    """
    query: str = Field(..., description="La requête originale de l'utilisateur")
    result: str = Field(..., description="Résultat de l'analyse")
    analysis_details: Dict[str, Any] = Field(default_factory=dict, description="Détails de l'analyse")
    error: Optional[str] = Field(None, description="Message d'erreur si l'analyse a échoué")

@data_analysis_router.post("/analyze")
async def data_analysis(
    query: str,
    file: Optional[UploadFile] = File(None),
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    max_retries: int = 2
) -> DataAnalysisResponse:
    """
    Endpoint pour effectuer une analyse de données via l'agent Data Analysis.
    
    Args:
        query (str): La requête d'analyse
        file (Optional[UploadFile]): Fichier de données à analyser
        model_id (Optional[str]): ID du modèle à utiliser
        user_id (Optional[str]): ID de l'utilisateur
        session_id (Optional[str]): ID de session
        max_retries (int): Nombre maximum de tentatives en cas d'échec
    
    Returns:
        DataAnalysisResponse: Réponse structurée de l'analyse
    """
    try:
        # Initialiser l'agent d'analyse de données
        data_analysis_agent = get_data_analysis_agent(
            model_id=model_id,
            user_id=user_id,
            session_id=session_id
        )
        
        # Gestion du fichier uploadé
        if file:
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(await file.read())
                temp_file_path = temp_file.name
            
            # Effectuer l'analyse avec le fichier
            result = data_analysis_agent.perform_data_analysis(query, data_source=temp_file_path)
            
            # Supprimer le fichier temporaire
            os.unlink(temp_file_path)
        else:
            # Analyse sans fichier
            result = data_analysis_agent.print_response(query, stream=False)
        
        # Nettoyer et structurer la réponse
        cleaned_result = clean_response(str(result))
        
        return DataAnalysisResponse(
            query=query,
            result=cleaned_result,
            analysis_details={
                "agent_name": data_analysis_agent.name,
                "model_used": data_analysis_agent.model.id
            }
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de données : {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Erreur lors de l'analyse : {str(e)}"
        )
