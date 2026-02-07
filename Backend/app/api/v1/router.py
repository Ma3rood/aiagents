from fastapi import APIRouter
from app.api.v1.endpoints import health, image_to_form, translation, category_embeddings

api_router = APIRouter()

# Include different endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(image_to_form.router, prefix="/image-to-form", tags=["image-to-form"], include_in_schema=False)
api_router.include_router(translation.router, prefix="/translation", tags=["translation"])
api_router.include_router(category_embeddings.router, prefix="/category-embeddings", tags=["category-embeddings"]) 