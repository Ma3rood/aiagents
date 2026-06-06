from fastapi import APIRouter
from app.api.v1.endpoints import health, image_to_form, translation, category_embeddings, motor_image_to_form, listing_description, listing_verification, listing_legitimacy

api_router = APIRouter()

# Include different endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(image_to_form.router, prefix="/image-to-form", tags=["image-to-form"])
api_router.include_router(translation.router, prefix="/translation", tags=["translation"])
api_router.include_router(category_embeddings.router, prefix="/category-embeddings", tags=["category-embeddings"])
api_router.include_router(motor_image_to_form.router, prefix="/motor-image-to-form", tags=["motor-image-to-form"])
api_router.include_router(listing_description.router, prefix="/listing-description", tags=["listing-description"])
api_router.include_router(listing_verification.router, prefix="/listing-verification", tags=["listing-verification"])
api_router.include_router(listing_legitimacy.router, prefix="/listing-legitimacy", tags=["listing-legitimacy"])