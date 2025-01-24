from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import time
import logging

from api.settings import api_settings
from api.routes.v1_router import v1_router

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create a FastAPI App

    Returns:
        FastAPI: FastAPI App
    """
    start_time = time.time()
    logger.info("Début de l'initialisation de l'application")

    # Create FastAPI App
    app: FastAPI = FastAPI(
        title=api_settings.title,
        version=api_settings.version,
        docs_url="/docs" if api_settings.docs_enabled else None,
        redoc_url="/redoc" if api_settings.docs_enabled else None,
        openapi_url="/openapi.json" if api_settings.docs_enabled else None,
    )

    # Add v1 router
    app.include_router(v1_router)

    # Add Middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    end_time = time.time()
    logger.info(f"Initialisation de l'application terminée en {end_time - start_time:.2f} secondes")

    return app


# Create FastAPI app
app = create_app()

__all__ = ["app"]