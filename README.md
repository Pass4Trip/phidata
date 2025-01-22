# Répertoire actif : /Users/vinh/Documents/phidata-1

# Architecture Multi-Agent avec Phidata

## Vue d'ensemble

Ce projet implémente une architecture multi-agent flexible utilisant le framework Phidata. L'objectif est de créer un système intelligent capable de router et de traiter différents types de requêtes.

## Agents Disponibles

### 1. Main Router Agent (`main_router_agent.py`)
- **Rôle**: Analyser les requêtes utilisateur et router vers l'agent spécialisé approprié
- **Capacités**: 
  - Analyse sémantique de la demande
  - Sélection dynamique de l'agent le plus adapté
  - Possibilité d'étendre les capacités avec de nouveaux agents

### 2. Web Search Agent (`web.py`)
- **Rôle**: Recherche d'informations sur le web
- **Outils**: 
  - DuckDuckGo Search
  - Recherche d'informations actuelles et récentes

### 3. API Knowledge Agent (`api_knowledge.py`)
- **Rôle**: Accès à des connaissances via des API
- **Capacités**:
  - Requêtes sur Wikipedia
  - Extensible à d'autres sources de connaissances

### 4. Data Analysis Agent (`data_analysis.py`)
- **Rôle**: Analyse approfondie de jeux de données
- **Capacités**:
  - Chargement de fichiers CSV/Excel
  - Analyse statistique
  - Génération d'insights

## Installation

```bash
# Installer les dépendances
uv pip install .
```

## Utilisation

```python
from agents.main_router_agent import process_user_request

# Exemple de requête
result = process_user_request("Trouve les dernières nouvelles sur l'IA")
```

## Extensibilité

L'architecture permet facilement d'ajouter de nouveaux agents spécialisés en suivant le modèle existant.

## Dépendances

- Phidata
- OpenAI
- DuckDuckGo Search
- Pandas
- NumPy

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

### Déploiement
Le déploiement se fait via le script `scripts/deploy.sh` qui automatise les étapes suivantes :
1. Construction de l'image Docker
2. Taggage de l'image
3. Envoi de l'image vers le registry
4. Préparation du VPS
5. Import de l'image sur le cluster Kubernetes
6. Redéploiement du pod

Commande de déploiement :
```bash
./scripts/deploy.sh
```

### Surveillance
Surveillez les logs du pod pour diagnostiquer les problèmes :
```bash
kubectl logs postgres-rabbitmq-listener-<pod-name>
```

### Dépannage
- Vérifiez que les ports PostgreSQL et RabbitMQ sont accessibles
- Vérifiez les logs pour des détails spécifiques sur les erreurs de connexion
