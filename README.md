# API de Recherche Web Intelligente

## Description
Cette API fournit un endpoint de recherche web intelligent qui peut répondre à diverses requêtes en utilisant DuckDuckGo et OpenAI.

## Fonctionnalités
- Recherche web en temps réel
- Support des requêtes mathématiques
- Support des requêtes conceptuelles
- Réponses structurées et lisibles

## Endpoint

### `/web/search`

#### Paramètres
- `query` (obligatoire) : La requête de recherche
- `model_id` (optionnel) : ID du modèle OpenAI à utiliser
- `user_id` (optionnel) : ID de l'utilisateur
- `session_id` (optionnel) : ID de session
- `max_retries` (optionnel, défaut: 2) : Nombre maximum de tentatives en cas d'échec

#### Exemples de requêtes

1. Calcul mathématique
```bash
curl "http://localhost:8001/web/search?query=20%20*%2014"
# Réponse : The result of 20 × 14 is 280
```

2. Requête conceptuelle
```bash
curl "http://localhost:8001/web/search?query=Qu%27est-ce%20que%20le%20holisme%20%3F"
# Réponse : Une explication détaillée du holisme
```

## Dépendances
- FastAPI
- OpenAI
- DuckDuckGo Search

## Configuration
- Assurez-vous d'avoir une clé API OpenAI configurée
- Installez les dépendances avec `uv pip install .`

## Démarrage
```bash
uvicorn app:app --reload --port 8001
```

## Limitations
- Dépend de la disponibilité de DuckDuckGo
- Nombre de résultats limité à 3 par défaut