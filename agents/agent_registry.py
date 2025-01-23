from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from phi.agent import Agent

@dataclass
class AgentMetadata:
    """Métadonnées pour un agent"""
    name: str
    description: str
    capabilities: List[str]
    keywords: List[str]
    priority_tasks: List[str]
    required_tools: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

class AgentRegistry:
    """Registre central pour gérer les agents"""
    
    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._metadata: Dict[str, AgentMetadata] = {}
        
    def register_agent(self, agent: Agent, metadata: AgentMetadata) -> None:
        """
        Enregistrer un nouvel agent avec ses métadonnées
        """
        self._agents[metadata.name] = agent
        self._metadata[metadata.name] = metadata
        
    def unregister_agent(self, agent_name: str) -> None:
        """
        Supprimer un agent du registre
        """
        self._agents.pop(agent_name, None)
        self._metadata.pop(agent_name, None)
        
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """
        Récupérer un agent par son nom
        """
        return self._agents.get(agent_name)
        
    def get_metadata(self, agent_name: str) -> Optional[AgentMetadata]:
        """
        Récupérer les métadonnées d'un agent
        """
        return self._metadata.get(agent_name)
        
    def list_agents(self) -> List[str]:
        """
        Lister tous les agents enregistrés
        """
        return list(self._agents.keys())
        
    def get_agent_capabilities(self, agent_name: str) -> List[str]:
        """
        Récupérer les capacités d'un agent
        """
        metadata = self._metadata.get(agent_name)
        return metadata.capabilities if metadata else []
        
    def update_performance_metrics(self, agent_name: str, metrics: Dict[str, float]) -> None:
        """
        Mettre à jour les métriques de performance d'un agent
        """
        if agent_name in self._metadata:
            self._metadata[agent_name].performance_metrics.update(metrics)
            
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """
        Trouver les agents ayant une capacité spécifique
        """
        return [
            name for name, metadata in self._metadata.items()
            if capability in metadata.capabilities
        ]
        
    def find_agents_by_keywords(self, keywords: List[str]) -> List[str]:
        """
        Trouver les agents correspondant à certains mots-clés
        """
        return [
            name for name, metadata in self._metadata.items()
            if any(keyword in metadata.keywords for keyword in keywords)
        ]
        
    def export_registry(self) -> str:
        """
        Exporter le registre au format JSON
        """
        registry_data = {
            name: {
                "metadata": {
                    "name": meta.name,
                    "description": meta.description,
                    "capabilities": meta.capabilities,
                    "keywords": meta.keywords,
                    "priority_tasks": meta.priority_tasks,
                    "required_tools": meta.required_tools,
                    "performance_metrics": meta.performance_metrics,
                    "config": meta.config
                }
            }
            for name, meta in self._metadata.items()
        }
        return json.dumps(registry_data, indent=2)
        
    def import_registry(self, registry_json: str) -> None:
        """
        Importer un registre depuis un JSON
        """
        registry_data = json.loads(registry_json)
        for name, data in registry_data.items():
            if name not in self._agents:
                continue
            
            metadata = data.get("metadata", {})
            self._metadata[name] = AgentMetadata(
                name=metadata.get("name", ""),
                description=metadata.get("description", ""),
                capabilities=metadata.get("capabilities", []),
                keywords=metadata.get("keywords", []),
                priority_tasks=metadata.get("priority_tasks", []),
                required_tools=metadata.get("required_tools", []),
                performance_metrics=metadata.get("performance_metrics", {}),
                config=metadata.get("config", {})
            )

# Créer une instance globale du registre
agent_registry = AgentRegistry()
