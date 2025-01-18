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

## PostgreSQL-RabbitMQ Listener

### Description
Le listener PostgreSQL-RabbitMQ est un composant crucial de notre infrastructure de données. Il permet de capturer et de transmettre en temps réel les insertions dans la table `web_searcher__memory` du schéma `ai` vers une file d'attente RabbitMQ.

### Fonctionnalités
- Écoute les insertions dans la table `web_searcher__memory`
- Crée un trigger PostgreSQL pour détecter les nouveaux enregistrements
- Convertit chaque nouvelle ligne en message JSON
- Publie les messages dans la file d'attente RabbitMQ `web_searcher_memory_queue`

### Prérequis
- Python 3.11+
- PostgreSQL
- RabbitMQ
- Dépendances Python (voir `listeners/requirements.txt`)

### Installation

1. Construire l'image Docker
```bash
cd listeners
docker build -t postgres-rabbitmq-listener:v1 .
```

2. Configurer les variables d'environnement
Créez un fichier `.env` avec les variables suivantes :
```
PG_HOST=votre_hôte_postgresql
PG_PORT=5432
PG_DATABASE=votre_base_de_données
PG_USER=votre_utilisateur
PG_PASSWORD=votre_mot_de_passe
RABBITMQ_HOST=votre_hôte_rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=votre_utilisateur
RABBITMQ_PASSWORD=votre_mot_de_passe
```

3. Déployer avec Kubernetes
```bash
kubectl apply -f postgres-rabbitmq-listener-deployment.yaml
```

### Configuration du Trigger
Un trigger PostgreSQL est automatiquement créé sur la table `ai.web_searcher__memory` pour capturer les insertions.

### Gestion des Erreurs
- Reconnexions automatiques en cas de perte de connexion
- Journalisation détaillée des erreurs
- Délais de nouvelle tentative configurables

### Surveillance
Surveillez les logs du pod pour diagnostiquer les problèmes :
```bash
kubectl logs postgres-rabbitmq-listener-<pod-name>
```

### Dépannage
- Vérifiez que les variables d'environnement sont correctes
- Assurez-vous que les ports PostgreSQL et RabbitMQ sont accessibles
- Vérifiez les logs pour des détails spécifiques sur les erreurs de connexion
