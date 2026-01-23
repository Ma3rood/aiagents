from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List
from app.services.openrouter import OpenRouterService
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ImageToFormRequest(BaseModel):
    image_urls: List[HttpUrl]
    selected_language: str


@router.post(
    "",
    response_model=Dict[str, Any],
    summary="Extract Form Data from Images",
    description="Convert product images to marketplace listing form data using AI vision models",
    response_description="Extracted form data including product details like title, description, price, category, etc."
)
async def image_to_form_agent(request: ImageToFormRequest):
    """
    Convert multiple images of the same product to marketplace listing form data using AI agent.
    
    This endpoint uses Qwen3 VL 30B A3B Instruct model via OpenRouter to analyze product images
    and extract structured form fields such as title, description, price, category, condition, brand, model, etc.
    
    **Features:**
    - Supports multiple images of the same product
    - Extracts comprehensive product information
    - Language-aware extraction based on selected language
    
    **Request:**
    - `image_urls`: List of image URLs (at least one required)
    - `selected_language`: Target language for extraction (e.g., "en", "ar")
    
    **Response:**
    - `status`: Success status
    - `image_urls`: List of processed image URLs
    - `image_count`: Number of images processed
    - `language`: Language used for extraction
    - `form_data`: Extracted product information (title, description, price, category, etc.)
    
    Args:
        request: ImageToFormRequest containing image_urls (list) and selected_language
        
    Returns:
        Dictionary containing the extracted form data with fields like:
        - title, description, price, category, condition, brand, model, etc.
        
    Raises:
        HTTPException: 400 if no image URLs provided, 500 for processing errors
    """
    logger.info(
        f"Image to form request - Language: {request.selected_language}, "
        f"Image count: {len(request.image_urls) if request.image_urls else 0}"
    )
    
    try:
        # Validate that at least one image URL is provided
        if not request.image_urls or len(request.image_urls) == 0:
            logger.warning("Image to form request rejected: No image URLs provided")
            raise HTTPException(status_code=400, detail="At least one image URL is required")
        
        logger.debug(f"Processing {len(request.image_urls)} image(s) for form extraction")
        
        # Initialize OpenRouter service
        openrouter_service = OpenRouterService()
        
        # Extract form fields from multiple images
        logger.info(f"Calling OpenRouter service for image form extraction in {request.selected_language}")
        form_data = await openrouter_service.extract_form_fields_from_images(
            image_urls=[str(url) for url in request.image_urls],
            language=request.selected_language
        )
        
        logger.info(
            f"Image to form extraction completed successfully - "
            f"Language: {request.selected_language}, Images processed: {len(request.image_urls)}"
        )
        logger.debug(f"Extracted form data keys: {list(form_data.keys()) if isinstance(form_data, dict) else 'N/A'}")
        
        return {
            "status": "success",
            "image_urls": [str(url) for url in request.image_urls],
            "image_count": len(request.image_urls),
            "language": request.selected_language,
            "form_data": form_data
        }
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Value error in image to form: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in image to form: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
