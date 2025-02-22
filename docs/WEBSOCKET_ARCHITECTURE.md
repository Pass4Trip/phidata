# Architecture du Chatbot WebSocket Phidata

## 🌐 Vue d'Ensemble du Système

Le système de chatbot WebSocket est conçu pour permettre des interactions temps réel entre un utilisateur et un agent conversationnel intelligent, en utilisant une architecture modulaire et extensible.

## 📦 Composants Principaux

### 1. Session Manager (`session_manager.py`)
#### Rôle Principal
Gère le cycle de vie des sessions utilisateur et maintient le contexte des conversations.

#### Fonctionnalités Clés
- Création de sessions utilisateur
- Gestion des agents actifs
- Conservation de l'historique de conversation

#### Méthodes Principales
- `create_session(user_id)` : Initialise une nouvelle session
- `get_current_agent(user_id)` : Récupère l'agent actuel
- `add_to_conversation_history(user_id, message, role)` : Enregistre les messages

### 2. WebSocket Router (`websocket_router.py`)
#### Rôle Principal
Point d'entrée pour les connexions WebSocket, routage et traitement des messages.

#### Fonctionnalités Clés
- Acceptation des connexions WebSocket
- Validation des messages
- Dispatch vers l'agent approprié
- Gestion des réponses

#### Points Critiques
- Utilise `websocket_session_manager` pour la gestion des sessions
- Transforme les requêtes en appels d'agent
- Gère les erreurs de connexion et de traitement

### 3. Agent Base (`agent_base.py`)
#### Rôle Principal
Définit la logique de traitement des messages et la génération de réponses.

#### Caractéristiques
- Basé sur un modèle de langage OpenAI
- Instructions prédéfinies
- Génération de réponses structurées (JSON)
- Possibilité de créer des widgets dynamiques

#### Configuration
- Modèle par défaut : `gpt-4o-mini`
- Mode de réponse : JSON
- Gestion de l'historique de conversation

### 4. Client WebSocket (`chatbot_websocket.py`)
#### Rôle Principal
Interface côté client pour la communication WebSocket.

#### Fonctionnalités
- Établissement de la connexion
- Envoi et réception de messages
- Surveillance de l'état de la connexion
- Mise à jour de l'interface utilisateur

## 🔄 Flux de Communication

1. **Initialisation**
   - Création d'une session utilisateur
   - Établissement de la connexion WebSocket
   - Initialisation de l'agent par défaut

2. **Interaction**
   - Utilisateur envoie un message
   - Message transmis via WebSocket
   - Routeur reçoit et valide le message
   - Agent traite le message
   - Réponse renvoyée à l'utilisateur

3. **Gestion de Session**
   - Historique de conversation conservé
   - Possibilité de changer d'agent dynamiquement
   - Widgets générés selon le contexte

## 🛠 Technologies Utilisées
- Python
- FastAPI (WebSocket)
- OpenAI GPT
- Panel (Interface)
- WebSockets
- SQLAlchemy (Potentiel stockage)

## 🔒 Sécurité et Performances
- Authentification par `user_id`
- Sessions isolées
- Gestion des erreurs
- Monitoring de connexion

## 🚀 Extensibilité
- Architecture modulaire
- Ajout facile de nouveaux agents
- Configuration dynamique des widgets
- Support multi-utilisateurs

## 📊 Métriques et Logging
- Logs détaillés des sessions
- Suivi des interactions
- Débogage facilité

## 🔍 Points d'Amélioration Potentiels
- Authentification robuste
- Gestion avancée des erreurs
- Scalabilité
- Persistance des sessions

## 📝 Notes Techniques
- Utilisation de WebSockets pour communication temps réel
- JSON comme format d'échange
- Séparation claire des responsabilités
