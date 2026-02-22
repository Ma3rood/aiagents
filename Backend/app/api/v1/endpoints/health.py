from fastapi import APIRouter
from pydantic import BaseModel
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    message: str


@router.get("", response_model=HealthResponse, summary="Health Check", description="Check the health status of the API service")
async def health_check():
    """
    Health check endpoint to verify that the API service is running and accessible.
    
    Returns:
        HealthResponse: Status and message indicating service health
    """
    logger.debug("Health check endpoint called")
    return {
        "status": "healthy",
        "message": "Service is running"
    } 