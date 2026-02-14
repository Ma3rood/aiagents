"""
Listing Description Generator endpoint.

POST /api/v1/listing-description

Accepts product images, form field values, and a target language.
Returns a market-appealing, customer-engaging listing description.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Any, Dict, List
from app.services.openrouter import OpenRouterService
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ListingDescriptionRequest(BaseModel):
    """Request body for the listing description generator."""
    image_urls: List[HttpUrl]
    form_fields: Dict[str, Any]
    selected_language: str


class ListingDescriptionResponse(BaseModel):
    """Response body containing the generated description."""
    status: str
    description: str
    language: str
    image_count: int


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ListingDescriptionResponse,
    summary="Generate Listing Description",
    description=(
        "Generate a market-appealing, customer-engaging product listing "
        "description from images, form field values, and a target language. "
        "The AI acts like an expert salesperson to craft compelling copy."
    ),
    response_description="Generated listing description in the selected language",
)
async def generate_listing_description(request: ListingDescriptionRequest):
    """
    Generate a compelling marketplace listing description.

    The endpoint uses product images and form field values as context to
    produce a persuasive, benefit-oriented description written in the
    selected language.

    **Request:**
    - ``image_urls``: List of product image URLs (at least one required)
    - ``form_fields``: Dictionary of listing form field key-value pairs
      (e.g. title, category, condition, price, attributes …)
    - ``selected_language``: Target language code (e.g. ``"en"``, ``"ar"``)

    **Response:**
    - ``status``: ``"success"``
    - ``description``: The generated listing description text
    - ``language``: Language code used
    - ``image_count``: Number of images processed
    """
    logger.info(
        f"Listing description request – "
        f"language={request.selected_language}, "
        f"images={len(request.image_urls)}, "
        f"fields={len(request.form_fields)}"
    )

    if not request.image_urls:
        raise HTTPException(
            status_code=400,
            detail="At least one image URL is required",
        )

    if not request.form_fields:
        raise HTTPException(
            status_code=400,
            detail="Form fields dictionary cannot be empty",
        )

    image_urls = [str(url) for url in request.image_urls]

    try:
        openrouter_service = OpenRouterService()
        description = await openrouter_service.generate_listing_description(
            image_urls=image_urls,
            form_fields=request.form_fields,
            language=request.selected_language,
        )

        logger.info(
            f"Listing description generated successfully – "
            f"language={request.selected_language}, "
            f"description_length={len(description)}"
        )

        return ListingDescriptionResponse(
            status="success",
            description=description,
            language=request.selected_language,
            image_count=len(image_urls),
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.error(f"Validation error in listing description: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(
            f"Unexpected error in listing description: {exc}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error generating listing description: {exc}",
        )
