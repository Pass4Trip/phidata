# Guide pour Ajouter un Nouvel Agent dans l'Orchestrateur

## Prérequis
- Python 3.10+
- Bibliothèque Phidata
- Accès à OpenAI API
- Environnement virtuel UV configuré

## Étapes Détaillées pour Ajouter un Nouvel Agent

### 1. Création du Fichier d'Agent
Créez un nouveau fichier dans `/agents/[nom_agent].py` avec la structure suivante :

```python
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.memory.db.postgres import PgMemoryDb
from phi.storage.agent.postgres import PgAgentStorage

from agents.agent_base import build_postgres_url  # Pour la connexion DB

def get_[nom_agent]_agent(
    model_id: str = "gpt-4o-mini",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
    **kwargs
) -> Agent:
    """
    Crée un agent spécialisé pour [description de la spécialisation].
    
    Args:
        model_id (str): Identifiant du modèle OpenAI
        user_id (Optional[str]): ID utilisateur
        session_id (Optional[str]): ID de session
        debug_mode (bool): Mode de débogage
    
    Returns:
        Agent: Agent spécialisé configuré
    """
    db_url = build_postgres_url()  # Utiliser la fonction de construction d'URL PostgreSQL
    
    [nom_agent]_agent = Agent(
        instructions=[
            "Instructions spécifiques pour l'agent",
            "Définir clairement son rôle et ses capacités"
        ],
        name="[Nom de l'Agent]",
        debug_mode=debug_mode,
        user_id=user_id,
        session_id=session_id,
        memory=AgentMemory(
            db=PgMemoryDb(table_name="agent_memories", db_url=db_url),
            create_user_memories=True,
            update_user_memories_after_run=True,
        ),
        storage=PgAgentStorage(table_name="agent_sessions", db_url=db_url),
    )
    
    return [nom_agent]_agent
```

### 2. Création de la Route FastAPI
Créez `/api/routes/[nom_agent]_router.py` :

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from agents.[nom_agent] import get_[nom_agent]_agent

[nom_agent]_router = APIRouter()

class [NomAgent]Request(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@[nom_agent]_router.post("/process")
async def process_[nom_agent]_request(request: [NomAgent]Request):
    try:
        agent = get_[nom_agent]_agent(
            user_id=request.user_id, 
            session_id=request.session_id
        )
        
        result = await agent.run(request.query)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3. Mise à Jour de l'Orchestrateur
Dans `/agents/orchestrator.py`, modifiez la méthode `_initialize_specialized_agents()` :

```python
# Importer le nouvel agent
from agents.[nom_agent] import get_[nom_agent]_agent

def _initialize_specialized_agents(
    self, 
    enable_web_agent: bool = True,
    enable_agent_base: bool = True,
    enable_[nom_agent]: bool = False,  # Nouveau paramètre
    **kwargs
):
    agents = {}
    
    # Autres agents existants...
    
    # Ajouter le nouvel agent
    if enable_[nom_agent]:
        try:
            [nom_agent]_agent = get_[nom_agent]_agent()
            agents_with_descriptions = {
                '[nom_agent]': {
                    'agent': [nom_agent]_agent,
                    'description': "Description détaillée de l'agent"
                }
            }
            agents.update(agents_with_descriptions)
            logger.info("✅ [Nom Agent] initialisé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation de [Nom Agent] : {e}")
    
    return agents
```

### 4. Mise à Jour du Routeur Principal
Dans `/api/routes/v1_router.py`, ajoutez la nouvelle route :

```python
from .web_router import web_router
from .[nom_agent]_router import [nom_agent]_router

v1_router.include_router([nom_agent]_router, prefix="/[nom_agent]", tags=["[Nom Agent]"])
```

### 5. Mise à Jour des Dépendances
Mettre à jour `pyproject.toml` si de nouvelles dépendances sont requises :

```toml
[tool.uv.sources]
# Ajouter les nouvelles dépendances si nécessaire
```

### 6. Installation et Activation
```bash
# Installer les dépendances
uv pip install .

# Lancer l'application
uvicorn api.main:app --reload
```

### Bonnes Pratiques
- Utilisez toujours `gpt-4o-mini` comme modèle par défaut
- Gérez les erreurs de manière robuste
- Utilisez le logging pour le débogage
- Documentez clairement les capacités de l'agent

### Conseils Supplémentaires
- Testez l'agent individuellement avant de l'intégrer
- Vérifiez la compatibilité avec l'architecture existante
- Assurez-vous que l'agent respecte les instructions générales du système

## Exemple Complet
Consultez les agents existants (`web.py`, `agent_base.py`) comme modèles.
