from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.config import settings
from app.core.logging_config import setup_logging, get_logger
from app.core.middleware import LoggingMiddleware
from app.core.exception_handler import (
    http_exception_handler,
    validation_exception_handler,
    starlette_exception_handler,
    general_exception_handler
)
from app.api.v1.router import api_router

# Initialize logging
setup_logging(
    log_dir=settings.LOG_DIR,
    log_level=settings.LOG_LEVEL,
    max_bytes=settings.LOG_MAX_BYTES,
    backup_count=settings.LOG_BACKUP_COUNT,
    enable_console=settings.LOG_ENABLE_CONSOLE
)

logger = get_logger(__name__)

# OpenAPI metadata for enhanced documentation
tags_metadata = [
    {
        "name": "health",
        "description": "Health check endpoints to monitor API status and service availability.",
    },
    {
        "name": "translation",
        "description": "AI-powered translation endpoints for translating listing fields to different languages while preserving context and formatting. Supports multiple listing types: Marketplace, Motors, Services, Jobs, and Property.",
    },
    {
        "name": "category-embeddings",
        "description": "Generate semantic descriptions and embeddings for all categories and store them in QDrant",
    },
    {
        "name": "motor-image-to-form",
        "description": "Motor Image-to-Form AI Agent. Analyzes vehicle/motor images and returns a pre-filled listing form with confidence scores, driven by CSV-defined schemas. Supports 13 motor categories: Cars, Motorbikes, Boats & marine, Trucks, and more.",
    },
    {
        "name": "image-to-form",
        "description": "Marketplace Image-to-Form AI Agent. Converts product images into pre-filled listing form data using semantic category search across 4000+ categories.",
    },
    {
        "name": "listing-description",
        "description": "Generate market-appealing, customer-engaging product listing descriptions from images and form field values.",
    },
    {
        "name": "listing-verification",
        "description": "Verify that product images match form field values with resemblance scores and per-field/image reports.",
    },
    {
        "name": "listing-legitimacy",
        "description": "Check listing legitimacy by flagging policy-violating content in images and text fields.",
    },
]

app = FastAPI(
    title="Ma3rood AI Agents API",
    description="""
    Backend API for Ma3rood AI Agents application.
    
    ## Features
    
    * **Translation**: Translate listing fields with context-aware AI translation
    * **Health Monitoring**: Check API status and service health
    
    ## API Documentation
    
    * **Swagger UI**: Interactive API documentation at `/docs`
    * **ReDoc**: Alternative API documentation at `/redoc`
    * **OpenAPI Schema**: JSON schema available at `/openapi.json`
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "Ma3rood AI Agents API Support",
    },
    license_info={
        "name": "Proprietary",
    },
)

# Request logging middleware (must be added before other middleware)
app.add_middleware(LoggingMiddleware)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, starlette_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

@app.on_event("startup")
async def startup_event():
    """Log application startup."""
    logger.info("Application starting up")
    logger.info(f"API prefix: {settings.API_V1_PREFIX}")
    logger.info(f"Debug mode: {settings.DEBUG}")

@app.on_event("shutdown")
async def shutdown_event():
    """Log application shutdown."""
    logger.info("Application shutting down")

@app.get("/", tags=["info"])
async def root():
    """
    Root endpoint providing API information and documentation links.
    
    Returns basic information about the API including version, documentation links, and available endpoints.
    """
    return {
        "message": "Welcome to Ma3rood AI Agents API",
        "version": "1.0.0",
        "docs": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
            "info": "/docs/info"
        },
        "api_prefix": settings.API_V1_PREFIX,
        "endpoints": {
            "health": f"{settings.API_V1_PREFIX}/health",
            "translation": f"{settings.API_V1_PREFIX}/translation",
            "category-embeddings": f"{settings.API_V1_PREFIX}/category-embeddings",
            "image_to_form": f"{settings.API_V1_PREFIX}/image-to-form",
            "motor_image_to_form": f"{settings.API_V1_PREFIX}/motor-image-to-form",
            "listing_description": f"{settings.API_V1_PREFIX}/listing-description",
            "listing_verification": f"{settings.API_V1_PREFIX}/listing-verification",
            "listing_legitimacy": f"{settings.API_V1_PREFIX}/listing-legitimacy",
        }
    }

@app.get("/docs/info", tags=["info"])
async def docs_info():
    """
    Get detailed API documentation information.
    
    Returns comprehensive information about available endpoints, request/response formats, and usage examples.
    """
    return {
        "api_name": "Ma3rood AI Agents API",
        "version": "1.0.0",
        "description": "Backend API for Ma3rood AI Agents application",
        "base_url": settings.API_V1_PREFIX,
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_schema": "/openapi.json"
        },
        "endpoints": {
            "health": {
                "path": f"{settings.API_V1_PREFIX}/health",
                "method": "GET",
                "description": "Check API health status",
                "response": {
                    "status": "healthy",
                    "message": "Service is running"
                }
            },
            "translation": {
                "path": f"{settings.API_V1_PREFIX}/translation",
                "method": "POST",
                "description": "Translate listing fields to target language using AI with context-aware translation based on listing type",
                "request_body": {
                    "fields": [
                        {
                            "field_name": "string",
                            "text": "string"
                        }
                    ],
                    "listing_details": "object (optional)",
                    "target_language": "string",
                    "listing_type": "string (optional, default: 'Marketplace') - One of: 'Marketplace', 'Motors', 'Services', 'Jobs', 'Property'",
                    "model": "string (optional)"
                },
                "response": {
                    "status": "success",
                    "target_language": "string",
                    "translated_fields": [
                        {
                            "field_name": "string",
                            "original_text": "string",
                            "translated_text": "string"
                        }
                    ]
                }
            }
        },
        "usage_examples": {
            "health_check": {
                "curl": f"curl -X GET http://localhost:8000{settings.API_V1_PREFIX}/health",
                "python": f"""
import requests
response = requests.get('http://localhost:8000{settings.API_V1_PREFIX}/health')
print(response.json())
"""
            },
            "translation": {
                "curl": f"""
curl -X POST http://localhost:8000{settings.API_V1_PREFIX}/translation \\
  -H "Content-Type: application/json" \\
  -d '{{"fields": [{{"field_name": "title", "text": "Product Title"}}], "target_language": "ar", "listing_type": "Marketplace"}}'
""",
                "python": f"""
import requests
response = requests.post(
    'http://localhost:8000{settings.API_V1_PREFIX}/translation',
    json={{
        "fields": [{{"field_name": "title", "text": "Product Title"}}],
        "target_language": "ar",
        "listing_type": "Marketplace"
    }}
)
print(response.json())
""",
                "listing_types": {
                    "Marketplace": "For general marketplace/product listings - focuses on product descriptions, features, and specifications",
                    "Motors": "For vehicle listings - focuses on vehicle specifications, make/model, mileage, and automotive terminology",
                    "Services": "For service listings - focuses on service descriptions, qualifications, and professional terminology",
                    "Jobs": "For job listings - focuses on job descriptions, requirements, and HR terminology",
                    "Property": "For real estate listings - focuses on property descriptions, specifications, and real estate terminology"
                }
            }
        }
    } 