from typing import List, Optional

from pydantic import field_validator, Field
from pydantic_settings import BaseSettings
from pydantic_core.core_schema import FieldValidationInfo


class ApiSettings(BaseSettings):
    """Paramètres de l'API configurables via des variables d'environnement."""

    # Informations de base de l'API
    title: str = "agent-api"
    version: str = "1.0"

    # Environnement d'exécution
    runtime_env: str = Field(
        default="dev", 
        description="Environnement d'exécution (dev, stg, prd)"
    )

    # Configuration de la documentation
    docs_enabled: bool = True

    # Configuration des CORS
    cors_origin_list: List[str] = Field(default_factory=list)

    @field_validator("runtime_env")
    @classmethod
    def validate_runtime_env(cls, runtime_env: str) -> str:
        """Validation simple de l'environnement d'exécution."""
        if runtime_env not in ["dev", "stg", "prd"]:
            raise ValueError(f"Environnement invalide : {runtime_env}")
        return runtime_env

    @field_validator("cors_origin_list", mode="before")
    @classmethod
    def configure_cors(cls, cors_list: Optional[List[str]], info: FieldValidationInfo) -> List[str]:
        """Configuration dynamique des origines CORS."""
        cors_list = cors_list or []
        
        # Ajouts de base
        base_cors = [
            "https://phidata.app", 
            "https://www.phidata.app"
        ]
        
        # Ajouts spécifiques au développement
        if info.data.get("runtime_env") == "dev":
            base_cors.extend([
                "http://localhost", 
                "http://localhost:3000",
                "http://127.0.0.1",
                "http://127.0.0.1:3000"
            ])
        
        # Fusionner et dédupliquer
        return list(set(cors_list + base_cors))


# Création de l'instance de configuration
api_settings = ApiSettings()