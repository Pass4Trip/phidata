# Guide de Création et d'Intégration d'un Agent Phidata

## 🚀 Architecture du Système d'Agents

### Composants Principaux
1. **Agent** : Logique métier et capacités spécifiques
2. **Routeur API** : Exposition des fonctionnalités via FastAPI
3. **Registre d'Agents** : Gestion centralisée des métadonnées

## 📋 Structure de Fichiers pour un Nouvel Agent

```
phidata-1/
│
├── agents/
│   ├── [nom_agent].py           # Définition de l'agent
│   └── agent_registry.py        # Registre global des agents
│
├── api/
│   └── routes/
│       └── [nom_agent]_router.py  # Routes API pour l'agent
│
└── requirements.toml             # Dépendances
```

## 🛠 Étape 1 : Création de l'Agent (`agents/[nom_agent].py`)

### Modèle de Base
```python
from phi.agent import Agent
from phi.tools import *

class MonAgent:
    def __init__(
        self, 
        model: Optional[str] = None,
        debug_mode: bool = False
    ):
        self.agent = Agent(
            model=model or "gpt-4-1106-preview",
            instructions=[
                "Instructions spécifiques pour l'agent"
            ],
            tools=[
                # Outils nécessaires
                PythonTools(),
                DuckDuckGo()
            ]
        )
    
    async def methode_principale(self, parametres):
        """Méthode principale de l'agent"""
        # Logique métier
```

## 🔗 Étape 2 : Création du Routeur API (`api/routes/[nom_agent]_router.py`)

### Modèle de Routeur
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.[nom_agent] import get_[nom_agent]

router = APIRouter(prefix="/[nom_agent]", tags=["[Nom Agent]"])

class RequeteModele(BaseModel):
    # Modèle de données d'entrée

@router.post("/endpoint")
async def endpoint_principal(requete: RequeteModele):
    try:
        agent = get_[nom_agent]()
        return await agent.methode_principale(requete)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 📝 Étape 3 : Enregistrement dans le Registre d'Agents

### Métadonnées de l'Agent
```python
agent_registry.register_agent(
    mon_agent,
    AgentMetadata(
        name="Nom de l'Agent",
        description="Description détaillée",
        capabilities=[
            "capability1",
            "capability2"
        ],
        keywords=[
            "mot1", 
            "mot2"
        ],
        priority_tasks=[
            "tâche1",
            "tâche2"
        ],
        required_tools=[
            "tool1", 
            "tool2"
        ]
    )
)
```

## 🔍 Paramètres des Métadonnées

### Capabilities
Liste des compétences spécifiques de l'agent.
- Utilisé pour le routage intelligent
- Permet le chaînage d'agents
- Contribue au scoring de sélection

### Keywords
Mots-clés déclenchant la sélection de l'agent.
- Matching textuel simple
- Aide à la sélection contextuelle

### Priority Tasks
Tâches principales pour lesquelles l'agent est optimisé.
- Guide la sélection de l'agent
- Permet une allocation de tâches plus précise

### Required Tools
Outils nécessaires au fonctionnement de l'agent.
- Vérifie la disponibilité des ressources
- Aide à la gestion des dépendances

## 🔧 Intégration Finale

1. Ajouter l'import dans `agents/orchestrator.py`
2. Initialiser l'agent dans `__init__`
3. Ajouter au registre dans `_register_default_agents()`
4. Mettre à jour `api/routes/v1_router.py`
5. Ajouter les dépendances dans `requirements.toml`

## 💡 Bonnes Pratiques

- Utilisez des noms de méthodes et de classes explicites
- Gérez les erreurs de manière robuste
- Documentez vos méthodes
- Utilisez des types de retour précis
- Testez unitairement chaque méthode

## 🚨 Conseils de Performance

- Limitez le nombre de requêtes externes
- Utilisez le caching quand possible
- Optimisez les méthodes longues
- Gérez les timeouts

## 📦 Dépendances

Utilisez `uv pip install` pour gérer les dépendances :

```bash
uv pip install -r requirements.txt
```

## 🔬 Debugging

- Activez le mode debug
- Utilisez des logs détaillés
- Capturez et loggez les exceptions

## 🤝 Contribution

1. Créez une branche dédiée
2. Documentez vos changements
3. Testez exhaustivement
4. Soumettez une pull request
