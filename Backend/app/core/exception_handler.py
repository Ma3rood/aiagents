from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.logging_config import get_logger

logger = get_logger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions and log them"""
    logger.error(
        f"HTTP Exception - Path: {request.url.path}, "
        f"Method: {request.method}, Status: {exc.status_code}, "
        f"Detail: {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors and log them"""
    errors = exc.errors()
    logger.warning(
        f"Validation Error - Path: {request.url.path}, "
        f"Method: {request.method}, Errors: {errors}"
    )
    return JSONResponse(
        status_code=422,
        content={"detail": errors, "body": exc.body}
    )


async def starlette_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle Starlette HTTP exceptions and log them"""
    logger.error(
        f"Starlette HTTP Exception - Path: {request.url.path}, "
        f"Method: {request.method}, Status: {exc.status_code}, "
        f"Detail: {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other unhandled exceptions and log them"""
    logger.critical(
        f"Unhandled Exception - Path: {request.url.path}, "
        f"Method: {request.method}, Error: {str(exc)}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
