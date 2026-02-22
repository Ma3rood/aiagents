"""
Listing Legitimacy Check endpoint.

POST /api/v1/listing-legitimacy

Accepts product images and form field values.
Returns a legitimacy report flagging policy-violating content
in both images (adult, violence, drugs, deceptive photos …)
and text fields (hate speech, scam signals, illegal goods …).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Any, Dict, List, Optional
from app.services.openrouter import OpenRouterService
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ListingLegitimacyRequest(BaseModel):
    """Request body for the listing legitimacy endpoint."""
    image_urls: List[HttpUrl]
    form_fields: Dict[str, Any]


class ImageFlag(BaseModel):
    """A single policy-violation flag raised against an image."""
    image_index: int
    category: str
    severity: str  # "critical" | "warning"
    description: str
    confidence: float


class ImageSummary(BaseModel):
    """Per-image legitimacy summary."""
    image_index: int
    is_legitimate: bool
    risk_level: str
    remark: str


class ImageAnalysis(BaseModel):
    """Aggregated image-legitimacy analysis."""
    is_legitimate: bool
    risk_level: str
    flags: List[ImageFlag]
    per_image_summary: List[ImageSummary]
    summary: Optional[str] = None
    error: Optional[str] = None


class TextFlag(BaseModel):
    """A single policy-violation flag raised against a text field."""
    field_name: str
    category: str
    severity: str  # "critical" | "warning"
    description: str
    matched_text: Optional[str] = None
    confidence: float


class FieldSummary(BaseModel):
    """Per-field legitimacy summary."""
    is_legitimate: bool
    risk_level: str
    remark: str


class TextAnalysis(BaseModel):
    """Aggregated text-legitimacy analysis."""
    is_legitimate: bool
    risk_level: str
    flags: List[TextFlag]
    per_field_summary: Dict[str, FieldSummary]
    summary: Optional[str] = None
    error: Optional[str] = None


class ListingLegitimacyResponse(BaseModel):
    """Full legitimacy report combining image and text analysis."""
    status: str
    is_legitimate: bool
    overall_risk_level: str
    total_flags: int
    image_analysis: ImageAnalysis
    text_analysis: TextAnalysis
    summary: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ListingLegitimacyResponse,
    summary="Check Listing Legitimacy",
    description=(
        "Inspect product images and listing form field values for "
        "policy violations (adult content, violence, drugs, scam signals, "
        "hate speech, illegal goods, misleading photos, etc.). "
        "Returns a structured report with per-image and per-field flags."
    ),
    response_description="Structured legitimacy report with flags, risk levels, and summaries",
)
async def check_listing_legitimacy(request: ListingLegitimacyRequest):
    """
    Run a comprehensive legitimacy check on a marketplace listing.

    The AI agent inspects every image for visual policy violations and
    every form-field value for textual policy violations, then merges
    the results into a single report.

    **Image checks** cover: adult / sexual content, minor safety,
    violence & weapons, drugs & controlled substances, illegal services,
    misleading / deceptive photos, and low-quality / spam images.

    **Text checks** cover: hate speech, adult solicitation, fraud / scam
    signals, misrepresentation, illegal goods or services, and personal
    data exposure.

    **Request:**
    - ``image_urls``: List of product image URLs (at least one required)
    - ``form_fields``: Dictionary of listing form field key-value pairs

    **Response:**
    - ``status``: ``"success"``
    - ``is_legitimate``: overall pass / fail
    - ``overall_risk_level``: ``"safe"`` | ``"warning"`` | ``"critical"``
    - ``total_flags``: total number of violations found
    - ``image_analysis``: per-image flags and summaries
    - ``text_analysis``: per-field flags and summaries
    - ``summary``: brief human-readable assessment
    """
    logger.info(
        f"Listing legitimacy request – "
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
        report = await openrouter_service.check_listing_legitimacy(
            image_urls=image_urls,
            form_fields=request.form_fields,
        )

        img = report.get("image_analysis", {})
        txt = report.get("text_analysis", {})

        logger.info(
            f"Listing legitimacy completed – "
            f"legitimate={report.get('is_legitimate')}, "
            f"risk={report.get('overall_risk_level')}, "
            f"flags={report.get('total_flags')}"
        )

        return ListingLegitimacyResponse(
            status=report.get("status", "success"),
            is_legitimate=report.get("is_legitimate", False),
            overall_risk_level=report.get("overall_risk_level", "critical"),
            total_flags=report.get("total_flags", 0),
            image_analysis=ImageAnalysis(
                is_legitimate=img.get("is_legitimate", False),
                risk_level=img.get("risk_level", "unknown"),
                flags=[ImageFlag(**f) for f in img.get("flags", [])],
                per_image_summary=[
                    ImageSummary(**s) for s in img.get("per_image_summary", [])
                ],
                summary=img.get("summary"),
                error=img.get("error"),
            ),
            text_analysis=TextAnalysis(
                is_legitimate=txt.get("is_legitimate", False),
                risk_level=txt.get("risk_level", "unknown"),
                flags=[TextFlag(**f) for f in txt.get("flags", [])],
                per_field_summary={
                    k: FieldSummary(**v)
                    for k, v in txt.get("per_field_summary", {}).items()
                },
                summary=txt.get("summary"),
                error=txt.get("error"),
            ),
            summary=report.get("summary", ""),
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.error(
            f"Validation error in listing legitimacy: {exc}", exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(
            f"Unexpected error in listing legitimacy: {exc}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error checking listing legitimacy: {exc}",
        )
