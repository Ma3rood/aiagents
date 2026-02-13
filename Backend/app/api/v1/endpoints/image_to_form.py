from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List, Optional
from app.services.openrouter import OpenRouterService
from app.services.qdrant_service import get_qdrant_service
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ImageToFormRequest(BaseModel):
    image_urls: List[HttpUrl]
    selected_language: str
    use_semantic_search: bool = True  # Flag to enable/disable semantic category search


class ImageToFormResponse(BaseModel):
    status: str
    image_urls: List[str]
    image_count: int
    language: str
    form_data: Dict[str, Any]


@router.post(
    "",
    response_model=Dict[str, Any],
    summary="Extract Form Data from Images",
    description="Convert product images to marketplace listing form data using AI vision models with semantic category search",
    response_description="Extracted form data including product details like description, category, condition, and attributes"
)
async def image_to_form_agent(request: ImageToFormRequest):
    """
    Convert multiple images of the same product to marketplace listing form data using AI agent.
    
    This endpoint uses a 3-step process:
    1. Extract visual description and determine root category from images
    2. Perform semantic search to find top 5 matching categories
    3. Select best category and generate description, condition, and attribute values
    
    **Features:**
    - Supports multiple images of the same product
    - Semantic category matching from 4000+ categories
    - Language-aware extraction based on selected language
    - Extracts relevant attributes based on category
    
    **Request:**
    - `image_urls`: List of image URLs (at least one required)
    - `selected_language`: Target language for extraction (e.g., "en", "ar")
    - `use_semantic_search`: Enable semantic category search (default: True)
    
    **Response:**
    - `status`: Success status
    - `image_urls`: List of processed image URLs
    - `image_count`: Number of images processed
    - `language`: Language used for extraction
    - `form_data`: Extracted product information including:
        - `description`: Product description in target language
        - `category`: Selected category with id_path and category_path
        - `condition`: Product condition (one of 7 predefined values)
        - `attributes`: Extracted attribute values
    
    Args:
        request: ImageToFormRequest containing image_urls (list) and selected_language
        
    Returns:
        Dictionary containing the extracted form data
        
    Raises:
        HTTPException: 400 if no image URLs provided, 500 for processing errors
    """
    logger.info(
        f"Image to form request - Language: {request.selected_language}, "
        f"Image count: {len(request.image_urls) if request.image_urls else 0}, "
        f"Semantic search: {request.use_semantic_search}"
    )
    
    try:
        # Validate that at least one image URL is provided
        if not request.image_urls or len(request.image_urls) == 0:
            logger.warning("Image to form request rejected: No image URLs provided")
            raise HTTPException(status_code=400, detail="At least one image URL is required")
        
        image_urls = [str(url) for url in request.image_urls]
        logger.debug(f"Processing {len(image_urls)} image(s) for form extraction")
        
        # Initialize services
        openrouter_service = OpenRouterService()
        
        if request.use_semantic_search:
            # Use 3-step semantic search flow
            form_data = await _process_with_semantic_search(
                openrouter_service=openrouter_service,
                image_urls=image_urls,
                language=request.selected_language
            )
        else:
            # Fall back to legacy single-call extraction
            form_data = await openrouter_service.extract_form_fields_from_images(
                image_urls=image_urls,
                language=request.selected_language
            )
        
        logger.info(
            f"Image to form extraction completed successfully - "
            f"Language: {request.selected_language}, Images processed: {len(image_urls)}"
        )
        logger.debug(f"Extracted form data keys: {list(form_data.keys()) if isinstance(form_data, dict) else 'N/A'}")
        
        return {
            "status": "success",
            "image_urls": image_urls,
            "image_count": len(image_urls),
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


async def _process_with_semantic_search(
    openrouter_service: OpenRouterService,
    image_urls: List[str],
    language: str
) -> Dict[str, Any]:
    """
    Process images using the 3-step semantic search flow.
    
    Step 1: Extract visual description and root category
    Step 2: Semantic search for top 5 categories
    Step 3: Select best category and generate output
    """
    
    # Step 1: Extract visual description and root category
    logger.info("Step 1: Extracting visual description and root category")
    visual_data = await openrouter_service.extract_visual_description_and_root_category(
        image_urls=image_urls
    )
    
    visual_description = visual_data["visual_description"]
    root_category = visual_data["root_category"]
    
    logger.info(f"Step 1 complete - Root category: {root_category}")
    logger.debug(f"Visual description: {visual_description[:200]}...")
    
    # Step 2: Semantic search for top 5 categories
    logger.info("Step 2: Performing semantic search for categories")
    
    try:
        qdrant_service = get_qdrant_service()
        
        # Create embedding for visual description
        query_embedding = await openrouter_service.create_embedding(visual_description)
        
        # Search with root category filter
        # candidate_categories = qdrant_service.search_categories(
        #     query_embedding=query_embedding,
        #     root_category_filter=root_category,
        #     top_k=5
        # )
        
        # If no results with filter, try without filter
        # if len(candidate_categories) == 0:
        logger.warning(f"Searching without root category filter")
        candidate_categories = qdrant_service.search_categories(
            query_embedding=query_embedding,
            root_category_filter=None,
            top_k=5
        )
        # else:
        #     logger.warning(f"Searching without root category filter")
        #     candidate_categories.extend(qdrant_service.search_categories(
        #         query_embedding=query_embedding,
        #         root_category_filter=None,
        #         top_k=5
        #     ))
        
        if not candidate_categories:
            logger.error("No candidate categories found in semantic search")
            raise ValueError("No matching categories found. Please ensure category embeddings are generated.")
        
        logger.info(f"Step 2 complete - Found {len(candidate_categories)} candidate categories")
        for i, cat in enumerate(candidate_categories):
            logger.debug(f"  {i+1}. {cat['category_path']} (score: {cat.get('score', 'N/A')})")
            
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        raise ValueError(f"Category search failed: {str(e)}. Ensure QDrant is running and embeddings are generated.")
    
    # Step 3: Select best category and generate output
    logger.info("Step 3: Selecting category and generating output")
    
    form_data = await openrouter_service.select_category_and_generate_output(
        image_urls=image_urls,
        visual_description=visual_description,
        candidate_categories=candidate_categories,
        language=language
    )
    
    logger.info(f"Step 3 complete - Selected category: {form_data.get('category', {}).get('category_path', 'N/A')}")
    
    return form_data


@router.post(
    "/legacy",
    response_model=Dict[str, Any],
    summary="Extract Form Data (Legacy)",
    description="Legacy endpoint that uses single-call extraction without semantic search"
)
async def image_to_form_legacy(request: ImageToFormRequest):
    """
    Legacy endpoint for image-to-form extraction without semantic category search.
    
    This uses the original single-call approach and is kept for backward compatibility.
    """
    logger.info(
        f"Legacy image to form request - Language: {request.selected_language}, "
        f"Image count: {len(request.image_urls) if request.image_urls else 0}"
    )
    
    try:
        if not request.image_urls or len(request.image_urls) == 0:
            raise HTTPException(status_code=400, detail="At least one image URL is required")
        
        image_urls = [str(url) for url in request.image_urls]
        openrouter_service = OpenRouterService()
        
        form_data = await openrouter_service.extract_form_fields_from_images(
            image_urls=image_urls,
            language=request.selected_language
        )
        
        return {
            "status": "success",
            "image_urls": image_urls,
            "image_count": len(image_urls),
            "language": request.selected_language,
            "form_data": form_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in legacy image to form: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
