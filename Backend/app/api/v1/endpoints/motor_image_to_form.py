"""
Motor Image-to-Form endpoint.

POST /api/v1/motor-image-to-form

Accepts vehicle image URLs and returns a confidence-scored, CSV-schema-driven
form pre-filled with visually observable data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Any, Dict, List, Optional
from app.services.motor_agent import MotorAgentService
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MotorImageToFormRequest(BaseModel):
    """Request body for the motor image-to-form endpoint."""
    image_urls: List[HttpUrl]
    combined_vision: bool = False  # True -> merge Stage 1+2 into one VLM call
    known_defects: Optional[List[str]] = None  # Seller-provided list of known defects to incorporate in description


class FieldOutput(BaseModel):
    value: Optional[Any] = None
    confidence: float = 0.0
    source: str = "image"
    needs_user_input: bool = True
    depends_on: Optional[str] = None
    required: Optional[bool] = None


class ImageQualityScore(BaseModel):
    """Quality assessment for a single image."""
    image_index: int
    score: float
    remark: str


class MotorImageToFormResponse(BaseModel):
    status: str
    category: str
    category_confidence: float
    fields: Dict[str, FieldOutput]
    image_urls: List[str]
    image_count: int
    image_quality_scores: List[ImageQualityScore] = []
    completed_stages: List[str]
    errors: List[str] = []


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=MotorImageToFormResponse,
    summary="Motor Image to Form",
    description=(
        "Analyze vehicle/motor images and return a pre-filled listing form "
        "driven by CSV-defined schemas. Uses a 6-stage AI pipeline: "
        "category detection, visual facts extraction, schema loading, "
        "field eligibility resolution, field value generation, and output assembly."
    ),
    response_description="Pre-filled motor listing form with confidence scores per field",
)
async def motor_image_to_form(request: MotorImageToFormRequest):
    """
    Convert vehicle/motor images into a marketplace listing form.

    **Pipeline stages:**
    1. Motor Category Detection (VLM) -- classify into one of 13 categories
    2. Visual Neutral Facts Extraction (VLM) -- observe without guessing
    3. CSV Schema Loader -- load fields, constraints, dependencies
    4. Field Eligibility Resolver -- topological ordering + dependency logic
    5. Field Value Generator (VLM + rules) -- fill fields with confidence
    6. Final Form Output Generator -- assemble frontend-ready JSON

    **Request:**
    - ``image_urls``: one or more image URLs of the same vehicle/item
    - ``combined_vision``: if *true*, stages 1+2 run in a single VLM call (faster)
    - ``known_defects``: optional list of known defects (strings) to incorporate in the description; defects are communicated clearly while keeping the listing attractive

    **Response:**
    - ``status``: ``"success"`` or ``"partial"`` (if a stage failed)
    - ``category``: detected motor category name
    - ``category_confidence``: 0.0-1.0
    - ``fields``: dict of field_name -> {value, confidence, source, needs_user_input, ...}
    - ``completed_stages``: list of stage names that executed successfully
    - ``errors``: list of error messages (empty on full success)
    """
    logger.info(
        f"Motor image-to-form request -- "
        f"images={len(request.image_urls)}, combined_vision={request.combined_vision}"
    )

    if not request.image_urls:
        raise HTTPException(status_code=400, detail="At least one image URL is required")

    image_urls = [str(url) for url in request.image_urls]

    try:
        agent = MotorAgentService()
        result = await agent.run(
            image_urls=image_urls,
            combined_vision=request.combined_vision,
            known_defects=request.known_defects,
        )

        logger.info(
            f"Motor image-to-form completed -- "
            f"status={result.status}, category={result.category}, "
            f"fields_filled={sum(1 for f in result.fields.values() if f.get('value') is not None)}"
        )

        return MotorImageToFormResponse(
            status=result.status,
            category=result.category,
            category_confidence=result.category_confidence,
            fields={
                name: FieldOutput(**data) for name, data in result.fields.items()
            },
            image_urls=image_urls,
            image_count=len(image_urls),
            image_quality_scores=[
                ImageQualityScore(**entry) for entry in result.image_quality_scores
            ],
            completed_stages=result.completed_stages,
            errors=result.errors,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.error(f"Validation error in motor image-to-form: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error in motor image-to-form: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing motor images: {exc}",
        )
