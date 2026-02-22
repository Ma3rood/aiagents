"""
Listing Verification endpoint.

POST /api/v1/listing-verification

Accepts product images and form field values.
Returns a verification report showing how well the images match
each form field and vice-versa, with resemblance scores and booleans.
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

class ListingVerificationRequest(BaseModel):
    """Request body for the listing verification endpoint."""
    image_urls: List[HttpUrl]
    form_fields: Dict[str, Any]


class ImageResult(BaseModel):
    """Verification result for a single image."""
    image_index: int
    matches_listing: bool
    resemblance_score: float
    remark: str


class FieldResult(BaseModel):
    """Verification result for a single form field."""
    field_value: Any
    matches_images: bool
    resemblance_score: float
    remark: str


class ImageQualityScore(BaseModel):
    """Quality assessment for a single image."""
    image_index: int
    score: float
    remark: str


class ListingVerificationResponse(BaseModel):
    """Response body containing the full verification report."""
    status: str
    image_count: int
    field_count: int
    image_results: List[ImageResult]
    field_results: Dict[str, FieldResult]
    image_quality_scores: List[ImageQualityScore]
    overall_match: bool
    overall_score: float
    summary: str


def _normalize_field_results(
    raw: Dict[str, Any],
) -> Dict[str, FieldResult]:
    """Build field_results from API response, tolerating list or dict per field."""
    required = {"field_value", "matches_images", "resemblance_score", "remark"}

    def _is_field_result_obj(d: Any) -> bool:
        return isinstance(d, dict) and required.issubset(d.keys())

    def _safe_field_result(d: Dict[str, Any]) -> Optional[FieldResult]:
        if not _is_field_result_obj(d):
            return None
        try:
            return FieldResult(**d)
        except Exception:
            return None

    out: Dict[str, FieldResult] = {}
    for k, v in raw.items():
        cand: Optional[Dict[str, Any]] = None
        if isinstance(v, dict):
            cand = v
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            cand = v[0]
        if cand is not None:
            result = _safe_field_result(cand)
            if result is not None:
                out[k] = result
            else:
                logger.warning(
                    "Listing verification: field_result for %s missing required keys or invalid, skipping",
                    k,
                )
        else:
            logger.warning(
                "Listing verification: invalid field_result for %s (type=%s), skipping",
                k,
                type(v).__name__,
            )
    return out


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ListingVerificationResponse,
    summary="Verify Listing Images vs Form Fields",
    description=(
        "Compare product images against listing form field values. "
        "Returns per-image and per-field resemblance scores with "
        "match booleans, plus an overall verdict."
    ),
    response_description="Structured verification report with scores and match flags",
)
async def verify_listing(request: ListingVerificationRequest):
    """
    Verify how well product images match the listing form field values.

    The AI agent inspects every image and every form field to produce
    two-way verification:

    1. **Per image** -- does each image match the described listing?
       (``matches_listing`` bool + ``resemblance_score`` 0-1)
    2. **Per form field** -- does each field value match what the images
       show? (``matches_images`` bool + ``resemblance_score`` 0-1)

    **Request:**
    - ``image_urls``: List of product image URLs (at least one required)
    - ``form_fields``: Dictionary of listing form field key-value pairs

    **Response:**
    - ``status``: ``"success"``
    - ``image_count`` / ``field_count``: counts processed
    - ``image_results``: list of per-image verdicts
    - ``field_results``: dict of per-field verdicts
    - ``image_quality_scores``: per-image quality assessments (score 0.0-1.0 and remark) indicating how well the product is understandable and identifiable from each image
    - ``overall_match``: True when >= 70% images AND fields match
    - ``overall_score``: weighted average score (0-1)
    - ``summary``: brief human-readable assessment
    """
    logger.info(
        f"Listing verification request – "
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
        report = await openrouter_service.verify_listing(
            image_urls=image_urls,
            form_fields=request.form_fields,
        )

        logger.info(
            f"Listing verification completed – "
            f"overall_match={report.get('overall_match')}, "
            f"overall_score={report.get('overall_score')}"
        )

        return ListingVerificationResponse(
            status="success",
            image_count=len(image_urls),
            field_count=len(request.form_fields),
            image_results=[
                ImageResult(**img) for img in report.get("image_results", [])
            ],
            field_results=_normalize_field_results(
                report.get("field_results", {})
            ),
            image_quality_scores=[
                ImageQualityScore(**entry)
                for entry in report.get("image_quality_scores", [])
            ],
            overall_match=report.get("overall_match") if report.get("overall_match") is not None else False,
            overall_score=report.get("overall_score") if report.get("overall_score") is not None else 0.0,
            summary=report.get("summary") or "",
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.error(
            f"Validation error in listing verification: {exc}", exc_info=True
        )
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(
            f"Unexpected error in listing verification: {exc}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error verifying listing: {exc}",
        )
