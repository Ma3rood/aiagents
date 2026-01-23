import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Thread lock for CSV file operations
_csv_lock = threading.Lock()


class AnalyticsService:
    """Service for logging analytics data to CSV files"""
    
    CSV_HEADERS = [
        "datetime",
        "agent_type",
        "model",
        "provider",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "cost",
        "input_char_count",
        "output_char_count",
        "time_taken_seconds",
        "target_language",
        "listing_type",
        "fields_count",
        "status",
        "error_message"
    ]
    
    def __init__(self, csv_file_path: str = "analytics/inference_analytics.csv"):
        """
        Initialize the analytics service.
        
        Args:
            csv_file_path: Path to the CSV file for storing analytics data
        """
        self.csv_file_path = Path(csv_file_path)
        self.csv_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_headers()
    
    def _ensure_headers(self):
        """Ensure CSV file exists with headers if it's a new file"""
        if not self.csv_file_path.exists():
            with _csv_lock:
                try:
                    with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                        writer.writeheader()
                    logger.info(f"Created analytics CSV file: {self.csv_file_path}")
                except Exception as e:
                    logger.error(f"Failed to create analytics CSV file: {str(e)}")
    
    def log_analytics(
        self,
        agent_type: str,
        model: str,
        provider: str = "OpenRouter",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost: float = 0.0,
        input_char_count: int = 0,
        output_char_count: int = 0,
        time_taken_seconds: float = 0.0,
        target_language: Optional[str] = None,
        listing_type: Optional[str] = None,
        fields_count: Optional[int] = None,
        status: str = "success",
        error_message: Optional[str] = None
    ):
        """
        Log analytics data to CSV file.
        
        Args:
            agent_type: Type of agent (e.g., "translation", "image_to_form")
            model: Model name used
            provider: Inference provider (e.g., "OpenRouter", "Anthropic", "OpenAI")
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens used
            cached_tokens: Number of cached tokens (if applicable)
            reasoning_tokens: Number of reasoning tokens (if applicable)
            cost: Cost in USD
            input_char_count: Total input character count
            output_char_count: Total output character count
            time_taken_seconds: Time taken in seconds
            target_language: Target language code (for translation)
            listing_type: Type of listing (Marketplace, Motors, etc.)
            fields_count: Number of fields processed
            status: Status of the operation (success, error, etc.)
            error_message: Error message if status is error
        """
        try:
            row = {
                "datetime": datetime.now().isoformat(),
                "agent_type": agent_type,
                "model": model,
                "provider": provider,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": cached_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cost": f"{cost:.10f}",  # Store with high precision
                "input_char_count": input_char_count,
                "output_char_count": output_char_count,
                "time_taken_seconds": f"{time_taken_seconds:.4f}",
                "target_language": target_language or "",
                "listing_type": listing_type or "",
                "fields_count": fields_count or "",
                "status": status,
                "error_message": error_message or ""
            }
            
            with _csv_lock:
                file_exists = self.csv_file_path.exists()
                with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
            
            logger.debug(f"Analytics logged: {agent_type} - {model} - {status}")
        except Exception as e:
            logger.error(f"Failed to log analytics to CSV: {str(e)}", exc_info=True)


# Global analytics service instance
_analytics_service: Optional[AnalyticsService] = None


def get_analytics_service(csv_file_path: Optional[str] = None) -> AnalyticsService:
    """
    Get or create the global analytics service instance.
    
    Args:
        csv_file_path: Optional path to CSV file (uses default if not provided)
        
    Returns:
        AnalyticsService instance
    """
    global _analytics_service
    if _analytics_service is None:
        if csv_file_path is None:
            from app.core.config import settings
            csv_file_path = settings.ANALYTICS_CSV_PATH
        _analytics_service = AnalyticsService(csv_file_path)
    return _analytics_service
