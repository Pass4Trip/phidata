"""
Prompts dédiés à l'agent Orchestrateur dans le framework Phidata

Ce module contient les modèles de prompts structurés pour guider 
l'agent orchestrateur dans la décomposition, l'exécution et la synthèse de tâches.
"""

from typing import Dict, Any

def get_task_decomposition_prompt(user_request: str) -> str:
    """
    Génère un prompt pour la décomposition de tâche
    
    Args:
        user_request (str): La requête originale de l'utilisateur
    
    Returns:
        str: Prompt structuré pour la décomposition de tâche
    """
    return f"""
Rôle: Agent Orchestrateur spécialisé dans la décomposition de tâches complexes

Objectif: Transformer une requête utilisateur en un plan structuré de sous-tâches précises

Instructions de décomposition:
1. Analyse approfondie de la requête originale
2. Identification des objectifs principaux
3. Décomposition en sous-tâches réalisables
4. Attribution des sous-tâches aux agents spécialisés

Agents disponibles:
- Web Search Agent: Recherches web, collecte d'informations actuelles
- API Knowledge Agent: Synthèse de connaissances via API
- Data Analysis Agent: Analyse de données, génération d'insights

Requête à décomposer: {user_request}

Format de réponse REQUIS (JSON strict):
{
    "task_name": "Nom descriptif de la tâche globale",
    "subtasks": [
        {
            "id": "identifiant_unique",
            "description": "Description précise de la sous-tâche",
            "suggested_agent": "Agent recommandé (Web/API/Data)",
            "priority": "haute/moyenne/basse"
        }
    ],
    "expected_outcome": "Résultat final attendu"
}

Conseils:
- Soyez précis et actionnable
- Anticipez les dépendances entre sous-tâches
- Proposez un plan flexible et adaptable
"""

def get_task_execution_prompt(
    original_request: str, 
    task_plan: Dict[str, Any], 
    partial_results: Dict[str, Any]
) -> str:
    """
    Génère un prompt pour l'évaluation de l'exécution des tâches
    
    Args:
        original_request (str): La requête originale de l'utilisateur
        task_plan (dict): Plan initial des tâches
        partial_results (dict): Résultats partiels des sous-tâches
    
    Returns:
        str: Prompt structuré pour l'évaluation de l'exécution
    """
    return f"""
Rôle: Superviseur de l'exécution des sous-tâches

Contexte:
- Requête originale: {original_request}
- Plan de tâches: {task_plan}
- Résultats partiels: {partial_results}

Mission:
1. Évaluer la progression des sous-tâches
2. Identifier les blocages potentiels
3. Proposer des ajustements stratégiques
4. Préparer la synthèse finale

Critères d'évaluation:
- Cohérence des résultats partiels
- Alignement avec l'objectif initial
- Qualité et pertinence des informations collectées

Action requise:
Produire un rapport structuré comprenant:
- Statut global de la mission
- Insights principaux
- Recommandations pour compléter ou affiner les résultats

Format de réponse REQUIS (JSON):
{{
    "mission_status": "en_cours/partiellement_complete/complete/bloquee",
    "key_insights": ["insight1", "insight2"],
    "recommendations": ["recommandation1", "recommandation2"],
    "final_synthesis": "Résumé concis et stratégique"
}}
"""

def get_task_synthesis_prompt(
    original_request: str, 
    task_results: Dict[str, Any]
) -> str:
    """
    Génère un prompt pour la synthèse finale des résultats
    
    Args:
        original_request (str): La requête originale de l'utilisateur
        task_results (dict): Résultats complets des sous-tâches
    
    Returns:
        str: Prompt structuré pour la synthèse des résultats
    """
    return f"""
Rôle: Agent de synthèse et d'intégration des connaissances

Contexte:
- Requête originale: {original_request}
- Résultats complets des sous-tâches: {task_results}

Mission:
1. Intégrer les résultats des différents agents
2. Produire une réponse cohérente et comprehensive
3. Mettre en évidence les points clés
4. Assurer la valeur ajoutée par rapport à la requête initiale

Directives de synthèse:
- Éliminer les redondances
- Hiérarchiser les informations
- Maintenir un niveau de détail pertinent
- Garantir la précision et la clarté

Format de réponse REQUIS:
{{
    "synthesized_response": "Réponse intégrée et structurée",
    "key_points": ["point1", "point2"],
    "sources": ["source1", "source2"],
    "confidence_level": "élevée/moyenne/faible"
}}
"""

TASK_DECOMPOSITION_PROMPT = """
Décompose la tâche suivante en sous-tâches plus petites et gérables.

Tâche principale : {user_request}

Instructions:
1. Analyse la tâche en détail
2. Identifie les sous-tâches nécessaires
3. Attribue une priorité à chaque sous-tâche

Format de réponse :
```json
{
    "sub_tasks": [
        {
            "description": "Description de la sous-tâche",
            "priority": "haute|moyenne|basse"

        }
    ]
}
```
"""

AGENT_ROUTING_PROMPT = """
Sélectionne l'agent le plus approprié pour la tâche suivante.

Tâche : {task}

Agents disponibles :
{agents_description}

Historique de performance des agents :
{performance_history}

Critères de sélection :
1. Correspondance des compétences avec la tâche
2. Historique de performance
3. Complexité de la tâche
4. Spécialisation de l'agent

Format de réponse :
```json
{
    "selected_agent": "Nom de l'agent sélectionné",
    "confidence_score": 0.95,
    "reasoning": "Explication détaillée du choix",
    "fallback_agent": "Nom de l'agent alternatif",
    "task_complexity": "simple|moyenne|complexe"
}
```
"""

TASK_CONTEXT_PROMPT = """
Analyse le contexte de la tâche pour améliorer la sélection de l'agent.

Tâche : {task}
Contexte actuel : {context}
Historique des tâches : {task_history}

Instructions :
1. Identifie les patterns et dépendances
2. Évalue la continuité avec les tâches précédentes
3. Détermine les contraintes et prérequis

Format de réponse :
```json
{
    "context_analysis": {
        "patterns": ["pattern1", "pattern2"],
        "dependencies": ["dep1", "dep2"],
        "constraints": ["constraint1", "constraint2"]
    },
    "recommended_approach": "Description de l'approche recommandée",
    "critical_considerations": ["consideration1", "consideration2"]
}
```
"""
