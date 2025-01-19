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

## Listener PostgreSQL-RabbitMQ

### Description
Un listener qui capture les insertions dans une table PostgreSQL et les publie sur une file d'attente RabbitMQ.

### Fonctionnalités
- Écoute dynamique des insertions sur une table PostgreSQL
- Transformation automatique des payloads
- Gestion robuste des connexions
- Journalisation détaillée

### Configuration
Le listener peut être configuré avec les paramètres suivants :
- `pg_schema`: Schéma PostgreSQL (défaut: 'ai')
- `table_name`: Nom de la table à surveiller (défaut: 'web_searcher__memory')
- `queue_name`: Nom de la file RabbitMQ (défaut: 'queue_vinh_test')
- `notification_channel`: Canal de notification PostgreSQL (défaut: 'web_searcher_memory_channel')

### Exemple d'Utilisation
```python
listener = PostgresRabbitMQListener(
    pg_schema='ai', 
    table_name='ma_table', 
    queue_name='ma_queue', 
    notification_channel='ma_notification_channel'
)
listener.start_listening()
```

### Transformation des Payloads
Le listener transforme automatiquement les payloads d'insertion :
- Opération : Toujours 'INSERT'
- Table source incluse
- Données de la nouvelle ligne capturées

### Gestion des Erreurs
- Reconnexions automatiques
- Journalisation des erreurs de connexion et de traitement
- Mécanisme de nouvelle tentative configurable

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

2. Déployer avec Kubernetes
```bash
kubectl apply -f postgres-rabbitmq-listener-deployment.yaml
```

### Surveillance
Surveillez les logs du pod pour diagnostiquer les problèmes :
```bash
kubectl logs postgres-rabbitmq-listener-<pod-name>
```

### Dépannage
- Vérifiez que les ports PostgreSQL et RabbitMQ sont accessibles
- Vérifiez les logs pour des détails spécifiques sur les erreurs de connexion
