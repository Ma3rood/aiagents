from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True

    # Logging Configuration
    LOG_DIR: str = "logs"
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    LOG_ENABLE_CONSOLE: bool = True
    
    # Analytics Configuration
    ANALYTICS_CSV_PATH: str = "analytics/inference_analytics.csv"
    
    # Category semantic descriptions CSV (output of Phase 1, input for Phase 2)
    CATEGORY_SEMANTIC_CSV_PATH: str = "applicaion_data/category_semantic_descriptions.csv"

    # OpenRouter Configuration
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_EMBEDDING_URL: str = "https://openrouter.ai/api/v1/embeddings"
    
    # QDrant Configuration
    # When Backend runs in Docker: use QDRANT_HOST=host.docker.internal to reach QDrant on host,
    # or the QDrant service/container name if both are on the same Docker network (e.g. qdrant).
    QDRANT_HOST: str = "host.docker.internal"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "Ma3roodAIAgentsMarketplaceCategories"

    # Motor CSV Configuration
    MOTOR_CATEGORIES_CSV_PATH: str = "applicaion_data/motor_categories.csv"
    MOTOR_CATEGORY_FIELDS_CSV_PATH: str = "applicaion_data/motor_category_fields.csv"
    MOTOR_FIELD_CONSTRAINTS_CSV_PATH: str = "applicaion_data/motor_field_constraints.csv"
    MOTOR_CONFIDENCE_THRESHOLD: float = 0.5

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings() 