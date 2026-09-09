import csv
import json
import asyncio
import os
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from app.api.deps import (
    TEXT_EMBEDDING_HEADERS_DOCS,
    OpenRouterHeaderOverrides,
    create_openrouter_service,
    get_text_embedding_overrides,
)
from app.services.openrouter import OpenRouterService
from app.services.qdrant_service import get_qdrant_service
from app.core.logging_config import get_logger
from app.core.config import settings

logger = get_logger(__name__)

router = APIRouter()

# Paths relative to Backend directory when running from project root
HIERARCHY_CSV_PATH = "applicaion_data/ma3rood_hierarchy.csv"

# Track embedding generation progress
embedding_progress: Dict[str, Any] = {
    "status": "idle",
    "phase": None,
    "total": 0,
    "processed": 0,
    "failed": 0,
    "current_category": None,
    "errors": []
}


class CategoryEmbeddingRequest(BaseModel):
    semantic_batch_size: int = Field(
        default=20,
        ge=1,
        description=(
            "Phase 1 batch size: how many category paths are sent together in a single "
            "text-model LLM call to generate semantic descriptions. Larger values reduce "
            "API round-trips but increase prompt size and failure blast radius for that batch."
        ),
        example=20,
    )
    embedding_batch_size: int = Field(
        default=50,
        ge=1,
        description=(
            "Phase 2 batch size: how many semantic-description rows are embedded together "
            "in one embeddings API call and then upserted to QDrant in the same batch. "
            "Larger values are faster but use more memory per request."
        ),
        example=50,
    )
    start_from: int = Field(
        default=0,
        ge=0,
        description="Phase 1 only: 0-based index into the hierarchy CSV to start from (for resume).",
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Phase 1 only: optional max number of categories to process after start_from.",
    )
    skip_phase1: bool = Field(
        default=False,
        description=(
            "If true, skip Phase 1 and run Phase 2 only using the existing semantic "
            "descriptions CSV. Useful when descriptions are already generated."
        ),
    )


GENERATE_EMBEDDINGS_DOCS = """
Generate category semantic descriptions and vector embeddings, then store them in QDrant.

This is a **long-running background job**. Poll `GET /category-embeddings/status` for progress.

## Phase 1 - Semantic descriptions

1. Loads all categories from the hierarchy CSV (`ma3rood_hierarchy.csv`).
2. Splits them into batches of `semantic_batch_size` (default **20**).
3. For each batch, calls the **text model** once to produce:
   - a `semantic_description` (what products belong in that category)
   - `relevant_attributes` for listings in that category
4. Appends results to the semantic descriptions CSV
   (`category_semantic_descriptions.csv`).

Use `start_from` / `limit` to process a subset, or `skip_phase1=true` to reuse an existing semantic CSV.

## Phase 2 - Embeddings + QDrant

1. Loads rows from the semantic descriptions CSV produced in Phase 1.
2. Splits them into batches of `embedding_batch_size` (default **50**).
3. For each batch:
   - calls the **embedding model** once for all descriptions in the batch
   - upserts the resulting vectors + metadata into the QDrant collection

## Batching summary

| Parameter | Phase | Default | Meaning |
|-----------|-------|---------|---------|
| `semantic_batch_size` | Phase 1 | 20 | Categories per text-LLM request |
| `embedding_batch_size` | Phase 2 | 50 | Descriptions per embedding request + QDrant upsert |

Phases run sequentially (Phase 1 then Phase 2), unless `skip_phase1` is set.
""" + TEXT_EMBEDDING_HEADERS_DOCS


class CategoryEmbeddingResponse(BaseModel):
    status: str
    message: str
    total_categories: int
    processed: int
    failed: int


def _resolve_csv_path(relative_path: str, for_writing: bool = False) -> str:
    """Resolve path: try Backend dir then cwd. If for_writing, ensure parent dir exists and return path even when file does not exist."""
    for base in ("Backend", ".", ""):
        full = os.path.join(base, relative_path) if base else relative_path
        if for_writing:
            d = os.path.dirname(full)
            if d:
                os.makedirs(d, exist_ok=True)
            return full
        if os.path.isfile(full):
            return full
    return relative_path


def load_categories_from_csv() -> List[Dict[str, str]]:
    """Load categories from the hierarchy CSV file"""
    categories = []
    csv_path = _resolve_csv_path(HIERARCHY_CSV_PATH)
    if not os.path.isfile(csv_path):
        csv_path = HIERARCHY_CSV_PATH
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_path = row.get('id_path', '')
                category_path = row.get('category_path', '')
                
                if id_path and category_path:
                    root_category = category_path.split(' > ')[0]
                    categories.append({
                        "id_path": id_path,
                        "category_path": category_path,
                        "root_category": root_category
                    })
        
        logger.info(f"Loaded {len(categories)} categories from CSV")
        return categories
        
    except FileNotFoundError:
        logger.error(f"Category CSV file not found: {csv_path}")
        raise HTTPException(status_code=500, detail=f"Category file not found: {csv_path}")
    except Exception as e:
        logger.error(f"Error loading categories from CSV: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading categories: {str(e)}")


SEMANTIC_CSV_FIELDS = ["id_path", "category_path", "root_category", "semantic_description", "relevant_attributes"]


def load_semantic_descriptions_from_csv() -> List[Dict[str, Any]]:
    """Load category semantic descriptions from CSV (output of Phase 1)."""
    csv_path = _resolve_csv_path(settings.CATEGORY_SEMANTIC_CSV_PATH)
    if not os.path.isfile(csv_path):
        return []
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            attrs = row.get("relevant_attributes", "[]")
            try:
                attrs_list = json.loads(attrs) if isinstance(attrs, str) else attrs
            except json.JSONDecodeError:
                attrs_list = []
            rows.append({
                "id_path": row.get("id_path", ""),
                "category_path": row.get("category_path", ""),
                "root_category": row.get("root_category", ""),
                "semantic_description": row.get("semantic_description", ""),
                "relevant_attributes": attrs_list
            })
    logger.info(f"Loaded {len(rows)} semantic descriptions from CSV")
    return rows


async def phase1_generate_semantic_descriptions_to_csv(
    categories: List[Dict[str, str]],
    openrouter: OpenRouterService,
    semantic_batch_size: int
) -> Dict[str, int]:
    """
    Phase 1: Generate semantic descriptions in batches and write to CSV.
    """
    global embedding_progress
    
    csv_path = _resolve_csv_path(settings.CATEGORY_SEMANTIC_CSV_PATH, for_writing=True)
    processed = 0
    failed = 0
    write_header = not os.path.isfile(csv_path)
    
    for start in range(0, len(categories), semantic_batch_size):
        batch = categories[start:start + semantic_batch_size]
        embedding_progress["current_category"] = batch[0]["category_path"] if batch else None
        
        try:
            category_paths = [c["category_path"] for c in batch]
            results = await openrouter.generate_category_semantic_descriptions_batch(category_paths)
            
            with open(csv_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=SEMANTIC_CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                    write_header = False
                for i, cat in enumerate(batch):
                    res = results[i] if i < len(results) else {"semantic_description": "", "relevant_attributes": []}
                    writer.writerow({
                        "id_path": cat["id_path"],
                        "category_path": cat["category_path"],
                        "root_category": cat["root_category"],
                        "semantic_description": res.get("semantic_description", ""),
                        "relevant_attributes": json.dumps(res.get("relevant_attributes", []), ensure_ascii=False)
                    })
                    processed += 1
                    embedding_progress["processed"] += 1
            
            await asyncio.sleep(0.2)
            
        except Exception as e:
            failed += len(batch)
            embedding_progress["failed"] += len(batch)
            error_msg = f"Batch failed (categories {start}-{start+len(batch)}): {str(e)}"
            logger.error(error_msg)
            embedding_progress["errors"].append(error_msg)
    
    return {"processed": processed, "failed": failed}


async def phase2_embeddings_and_qdrant(
    qdrant_service,
    openrouter: OpenRouterService,
    embedding_batch_size: int
) -> Dict[str, int]:
    """
    Phase 2: Load semantic descriptions from CSV, generate embeddings in batch, upsert to QDrant in batch.
    """
    global embedding_progress
    
    rows = load_semantic_descriptions_from_csv()
    if not rows:
        logger.warning("No semantic descriptions in CSV; run Phase 1 first")
        return {"processed": 0, "failed": 0}
    
    embedding_progress["total"] = len(rows)
    embedding_progress["processed"] = 0
    processed = 0
    failed = 0
    
    for start in range(0, len(rows), embedding_batch_size):
        batch = rows[start:start + embedding_batch_size]
        embedding_progress["current_category"] = batch[0]["category_path"] if batch else None
        
        try:
            texts = [r["semantic_description"] for r in batch]
            embeddings = await openrouter.create_embeddings_batch(texts)
            
            batch_data = []
            for i, row in enumerate(batch):
                emb = embeddings[i] if i < len(embeddings) else None
                if emb is None:
                    failed += 1
                    continue
                batch_data.append({
                    "id_path": row["id_path"],
                    "category_path": row["category_path"],
                    "root_category": row["root_category"],
                    "semantic_description": row["semantic_description"],
                    "relevant_attributes": row["relevant_attributes"],
                    "embedding_vector": emb
                })
            
            if batch_data:
                qdrant_service.upsert_category_embeddings_batch(batch_data)
                processed += len(batch_data)
                embedding_progress["processed"] += len(batch_data)
            
            await asyncio.sleep(0.05)
            
        except Exception as e:
            failed += len(batch)
            embedding_progress["failed"] += len(batch)
            error_msg = f"Embedding batch failed (rows {start}-{start+len(batch)}): {str(e)}"
            logger.error(error_msg)
            embedding_progress["errors"].append(error_msg)
    
    return {"processed": processed, "failed": failed}


async def generate_embeddings_task(
    categories: List[Dict[str, str]],
    semantic_batch_size: int,
    embedding_batch_size: int,
    skip_phase1: bool,
    openrouter_overrides: Optional[OpenRouterHeaderOverrides] = None,
):
    """
    Background task: Phase 1 = generate semantic descriptions in batches and write to CSV;
    Phase 2 = load CSV, generate embeddings in batch, upsert to QDrant in batch.
    """
    global embedding_progress
    
    embedding_progress["status"] = "running"
    embedding_progress["total"] = len(categories)
    embedding_progress["processed"] = 0
    embedding_progress["failed"] = 0
    embedding_progress["errors"] = []
    
    try:
        openrouter = create_openrouter_service(openrouter_overrides)
        qdrant_service = get_qdrant_service()
        qdrant_service.ensure_collection_exists()
        
        if not skip_phase1:
            embedding_progress["phase"] = "semantic_descriptions"
            logger.info(f"Phase 1: Generating semantic descriptions for {len(categories)} categories (batch size: {semantic_batch_size})")
            result1 = await phase1_generate_semantic_descriptions_to_csv(
                categories=categories,
                openrouter=openrouter,
                semantic_batch_size=semantic_batch_size
            )
            logger.info(f"Phase 1 done - Processed: {result1['processed']}, Failed: {result1['failed']}")
        else:
            embedding_progress["total"] = len(load_semantic_descriptions_from_csv())
        
        embedding_progress["phase"] = "embeddings_qdrant"
        embedding_progress["processed"] = 0
        embedding_progress["failed"] = 0
        logger.info(f"Phase 2: Generating embeddings and upserting to QDrant (batch size: {embedding_batch_size})")
        result2 = await phase2_embeddings_and_qdrant(
            qdrant_service=qdrant_service,
            openrouter=openrouter,
            embedding_batch_size=embedding_batch_size
        )
        
        embedding_progress["status"] = "completed"
        embedding_progress["phase"] = None
        embedding_progress["current_category"] = None
        
        logger.info(
            f"Embedding generation completed - "
            f"Phase 2 Processed: {result2['processed']}, Failed: {result2['failed']}"
        )
        
    except Exception as e:
        embedding_progress["status"] = "failed"
        embedding_progress["phase"] = None
        embedding_progress["errors"].append(str(e))
        logger.error(f"Embedding generation task failed: {str(e)}", exc_info=True)


@router.post(
    "/generate",
    response_model=CategoryEmbeddingResponse,
    summary="Generate Category Embeddings",
    description=GENERATE_EMBEDDINGS_DOCS,
)
async def generate_category_embeddings(
    request: CategoryEmbeddingRequest,
    background_tasks: BackgroundTasks,
    openrouter_overrides: OpenRouterHeaderOverrides = Depends(get_text_embedding_overrides),
):
    """
    Generate embeddings for all categories from the hierarchy CSV (background job).

    ## Phase 1 - Semantic descriptions
    Loads categories from the hierarchy CSV and, in batches of `semantic_batch_size`
    (default 20), calls the text model to generate a semantic description and relevant
    attributes for each category. Results are appended to the semantic descriptions CSV.

    ## Phase 2 - Embeddings + QDrant
    Loads the semantic CSV and, in batches of `embedding_batch_size` (default 50),
    creates embedding vectors and upserts them into QDrant.

    ## Batching
    - **semantic_batch_size**: categories per Phase 1 text-LLM call
    - **embedding_batch_size**: descriptions per Phase 2 embedding call + QDrant upsert

    Use `/status` to monitor progress. Set `skip_phase1=true` to run Phase 2 only.
    """
    global embedding_progress
    
    if embedding_progress["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Embedding generation is already in progress. Check /status for progress."
        )
    
    categories = load_categories_from_csv()
    
    if request.start_from > 0:
        categories = categories[request.start_from:]
    if request.limit:
        categories = categories[:request.limit]
    
    if not categories and not request.skip_phase1:
        raise HTTPException(status_code=400, detail="No categories to process")
    
    if request.skip_phase1 and not os.path.isfile(_resolve_csv_path(settings.CATEGORY_SEMANTIC_CSV_PATH)):
        raise HTTPException(
            status_code=400,
            detail="skip_phase1 is true but semantic descriptions CSV not found. Run Phase 1 first."
        )
    
    logger.info(
        f"Starting embedding generation - Phase 1 batch: {request.semantic_batch_size}, "
        f"Phase 2 batch: {request.embedding_batch_size}, Categories: {len(categories)}, "
        f"Skip Phase 1: {request.skip_phase1}"
    )
    
    background_tasks.add_task(
        generate_embeddings_task,
        categories=categories,
        semantic_batch_size=request.semantic_batch_size,
        embedding_batch_size=request.embedding_batch_size,
        skip_phase1=request.skip_phase1,
        openrouter_overrides=openrouter_overrides,
    )
    
    total = len(load_semantic_descriptions_from_csv()) if request.skip_phase1 else len(categories)
    return CategoryEmbeddingResponse(
        status="started",
        message=f"Embedding generation started (Phase 1: {len(categories)} categories, Phase 2 from CSV)" if not request.skip_phase1 else "Embedding generation started (Phase 2 only, from existing CSV)",
        total_categories=total,
        processed=0,
        failed=0
    )


@router.get(
    "/status",
    response_model=Dict[str, Any],
    summary="Get Embedding Generation Status",
    description="Get the current status of the embedding generation process"
)
async def get_embedding_status():
    """
    Get the current status of the embedding generation process.
    
    **Returns:**
    - `status`: Current status (idle, running, completed, failed)
    - `total`: Total number of categories to process
    - `processed`: Number of categories processed so far
    - `failed`: Number of categories that failed
    - `current_category`: Currently processing category
    - `errors`: List of error messages (last 10)
    """
    return {
        "status": embedding_progress["status"],
        "phase": embedding_progress["phase"],
        "total": embedding_progress["total"],
        "processed": embedding_progress["processed"],
        "failed": embedding_progress["failed"],
        "current_category": embedding_progress["current_category"],
        "progress_percentage": (
            round(embedding_progress["processed"] / embedding_progress["total"] * 100, 2)
            if embedding_progress["total"] > 0 else 0
        ),
        "errors": embedding_progress["errors"][-10:]
    }


@router.get(
    "/collection-info",
    response_model=Dict[str, Any],
    summary="Get QDrant Collection Info",
    description="Get information about the category embeddings collection in QDrant"
)
async def get_collection_info():
    """Get information about the QDrant collection"""
    try:
        qdrant_service = get_qdrant_service()
        return qdrant_service.get_collection_info()
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/reset",
    response_model=Dict[str, str],
    summary="Reset Embedding Collection",
    description="Delete and recreate the category embeddings collection (USE WITH CAUTION)"
)
async def reset_collection():
    """
    Delete and recreate the QDrant collection.
    
    **WARNING:** This will delete all existing embeddings.
    """
    global embedding_progress
    
    if embedding_progress["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Cannot reset while embedding generation is in progress"
        )
    
    try:
        qdrant_service = get_qdrant_service()
        
        # Delete existing collection
        qdrant_service.delete_collection()
        
        # Recreate collection
        qdrant_service.ensure_collection_exists()
        
        # Reset progress
        embedding_progress["status"] = "idle"
        embedding_progress["phase"] = None
        embedding_progress["total"] = 0
        embedding_progress["processed"] = 0
        embedding_progress["failed"] = 0
        embedding_progress["current_category"] = None
        embedding_progress["errors"] = []
        
        logger.warning("Category embeddings collection has been reset")
        
        return {"status": "success", "message": "Collection has been reset"}
        
    except Exception as e:
        logger.error(f"Error resetting collection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
