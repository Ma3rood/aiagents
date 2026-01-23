import time
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming HTTP requests and responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timer
        start_time = time.time()
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Get request details
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        
        # Log request
        logger.info(
            f"Incoming request - Method: {method}, Path: {path}, "
            f"IP: {client_ip}, QueryParams: {query_params}"
        )
        
        # Read request body for POST/PUT/PATCH requests
        request_body = None
        body_bytes = None
        if method in ["POST", "PUT", "PATCH"]:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    # Try to parse as JSON for better logging
                    try:
                        request_body = json.loads(body_bytes.decode('utf-8'))
                        # Log request body (truncate if too long)
                        body_str = json.dumps(request_body, ensure_ascii=False)
                        if len(body_str) > 1000:
                            logger.debug(
                                f"Request body (truncated): {body_str[:1000]}..."
                            )
                        else:
                            logger.debug(f"Request body: {body_str}")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # If not JSON, log as text (truncate if too long)
                        body_str = body_bytes.decode('utf-8', errors='ignore')
                        if len(body_str) > 1000:
                            logger.debug(f"Request body (text, truncated): {body_str[:1000]}...")
                        else:
                            logger.debug(f"Request body (text): {body_str}")
            except Exception as e:
                logger.warning(f"Error reading request body: {str(e)}")
        
        # Restore request body if it was read
        if body_bytes is not None:
            async def receive():
                return {"type": "http.request", "body": body_bytes}
            request._receive = receive
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Get response status
            status_code = response.status_code
            
            # Log response
            logger.info(
                f"Response - Method: {method}, Path: {path}, "
                f"Status: {status_code}, ProcessTime: {process_time:.3f}s, IP: {client_ip}"
            )
            
            # Log error responses with more detail
            if status_code >= 400:
                logger.warning(
                    f"Error response - Method: {method}, Path: {path}, "
                    f"Status: {status_code}, ProcessTime: {process_time:.3f}s, IP: {client_ip}"
                )
            
            return response
            
        except Exception as e:
            # Calculate processing time even on error
            process_time = time.time() - start_time
            
            # Log exception
            logger.error(
                f"Unhandled exception - Method: {method}, Path: {path}, "
                f"Error: {str(e)}, ProcessTime: {process_time:.3f}s, IP: {client_ip}",
                exc_info=True
            )
            raise
