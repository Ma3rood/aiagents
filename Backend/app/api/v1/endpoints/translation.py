from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, Literal
from app.services.openrouter import OpenRouterService
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


class TranslationField(BaseModel):
    field_name: str
    text: str


class TranslationRequest(BaseModel):
    fields: list[TranslationField]
    listing_details: Optional[Dict[str, Any]] = None
    target_language: str
    listing_type: Literal["Marketplace", "Motors", "Services", "Jobs", "Property"] = "Marketplace"
    model: Optional[str] = None


class TranslatedField(BaseModel):
    field_name: str
    original_text: str
    translated_text: str


class TranslationResponse(BaseModel):
    status: str
    target_language: str
    translated_fields: list[TranslatedField]


@router.post(
    "",
    response_model=TranslationResponse,
    summary="Translate Listing Fields",
    description="Translate listing fields to target language using context-aware AI translation. Supports multiple listing types (Marketplace, Motors, Services, Jobs, Property) with type-specific terminology and context.",
    response_description="Translated fields with original and translated text for each field"
)
async def translate_listing_fields(request: TranslationRequest):
    """
    Translate listing fields to the target language using AI agent.
    
    This endpoint uses context-aware translation that preserves meaning, formatting, and terminology specific to the listing type.
    It adapts translation style and terminology based on the listing type (Marketplace, Motors, Services, Jobs, or Property).
    
    **Features:**
    - Context-aware translation using listing details and listing type
    - Type-specific terminology (e.g., automotive terms for Motors, real estate terms for Property)
    - Preserves field names and structure
    - Maintains formatting and special characters
    - Supports multiple fields in a single request
    
    **Request:**
    - `fields`: List of field objects with `field_name` and `text` to translate
    - `listing_details`: Optional context (category, product type, etc.) for better translation
    - `target_language`: Target language code (e.g., "ar", "en", "fr")
    - `listing_type`: Type of listing - Marketplace, Motors, Services, Jobs, or Property (defaults to Marketplace)
    - `model`: Optional AI model name (defaults to Qwen 3)
    
    **Response:**
    - `status`: Success status
    - `target_language`: Language code used for translation
    - `translated_fields`: List of translated fields with original and translated text
    
    Args:
        request: TranslationRequest containing:
            - fields: List of field names with their text to translate
            - listing_details: Optional context about the listing (category, product type, etc.)
            - target_language: The language to translate to
            - listing_type: Type of listing - Marketplace, Motors, Services, Jobs, or Property
            - model: Optional model name to use (defaults to Qwen 3)
        
    Returns:
        TranslationResponse with translated fields preserving field names
        
    Raises:
        HTTPException: 400 if no fields provided, 500 for translation errors
    """
    logger.info(
        f"Translation request - Target language: {request.target_language}, "
        f"Listing type: {request.listing_type}, "
        f"Fields count: {len(request.fields) if request.fields else 0}, "
        f"Model: {request.model or 'default'}"
    )
    
    try:
        if not request.fields or len(request.fields) == 0:
            logger.warning("Translation request rejected: No fields provided")
            raise HTTPException(status_code=400, detail="At least one field is required for translation")
        
        # Initialize OpenRouter service
        openrouter_service = OpenRouterService()
        
        # Prepare fields for translation
        fields_to_translate = {field.field_name: field.text for field in request.fields}
        logger.debug(f"Fields to translate: {list(fields_to_translate.keys())}")
        
        # Translate using OpenRouter
        logger.info(f"Calling OpenRouter service for translation to {request.target_language}")
        translated_data = await openrouter_service.translate_listing_fields(
            fields=fields_to_translate,
            listing_details=request.listing_details,
            target_language=request.target_language,
            listing_type=request.listing_type,
            model=request.model
        )
        
        # Build response
        translated_fields = [
            TranslatedField(
                field_name=field.field_name,
                original_text=field.text,
                translated_text=translated_data.get(field.field_name, field.text)
            )
            for field in request.fields
        ]
        
        logger.info(
            f"Translation completed successfully - Target language: {request.target_language}, "
            f"Translated fields: {len(translated_fields)}"
        )
        
        return TranslationResponse(
            status="success",
            target_language=request.target_language,
            translated_fields=translated_fields
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Value error in translation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in translation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error translating fields: {str(e)}")
