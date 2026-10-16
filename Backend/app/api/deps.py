"""Shared FastAPI dependencies for request-scoped configuration overrides."""

from dataclasses import dataclass
from typing import Optional

from fastapi import Header

from app.services.openrouter import OpenRouterService


@dataclass(frozen=True)
class OpenRouterHeaderOverrides:
    """Optional OpenRouter settings supplied via request headers."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    embedding_url: Optional[str] = None
    model: Optional[str] = None
    text_model: Optional[str] = None
    embedding_model: Optional[str] = None
    # Default False: thinking off unless the client explicitly enables it
    enable_text_thinking: bool = False


def _first_non_empty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _parse_bool_header(value: Optional[str], default: bool = False) -> bool:
    """Parse a boolean request header; empty/missing → default (False)."""
    if value is None or not str(value).strip():
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _headers_docs(*rows: tuple[str, str]) -> str:
    """Build a short OpenAPI note listing only the headers used by an endpoint."""
    lines = [
        "",
        "### Optional OpenRouter override headers",
        "",
        "Sent header values take precedence over `.env` defaults for this request.",
        "",
        "| Header | Overrides |",
        "|--------|-----------|",
    ]
    for name, purpose in rows:
        lines.append(f"| `{name}` | {purpose} |")
    return "\n".join(lines)


# Endpoint-specific docs snippets (only headers that endpoint uses)
VISION_HEADERS_DOCS = _headers_docs(
    ("X-Api-Key", "API key"),
    ("X-Base-Url", "Chat completions URL"),
    ("X-Model", "Vision model"),
)

LISTING_DESCRIPTION_HEADERS_DOCS = _headers_docs(
    ("X-Api-Key", "API key"),
    ("X-Base-Url", "Chat completions URL"),
    ("X-Model", "Vision model"),
    (
        "X-Enable-Text-Thinking",
        "Enable model thinking on text-only path (default false; set true to enable)",
    ),
)

TEXT_HEADERS_DOCS = _headers_docs(
    ("X-Api-Key", "API key"),
    ("X-Base-Url", "Chat completions URL"),
    ("X-Model / X-Text-Model", "Text model (header wins over body `model`)"),
    (
        "X-Enable-Text-Thinking",
        "Enable model thinking (default false; set true to enable)",
    ),
)

VISION_EMBEDDING_HEADERS_DOCS = _headers_docs(
    ("X-Api-Key", "API key"),
    ("X-Base-Url", "Chat completions URL"),
    ("X-Embedding-Url", "Embeddings URL"),
    ("X-Model", "Vision model"),
    ("X-Embedding-Model", "Embedding model"),
)

TEXT_EMBEDDING_HEADERS_DOCS = _headers_docs(
    ("X-Api-Key", "API key"),
    ("X-Base-Url", "Chat completions URL"),
    ("X-Embedding-Url", "Embeddings URL"),
    ("X-Text-Model", "Text model (semantic descriptions)"),
    ("X-Embedding-Model", "Embedding model"),
    (
        "X-Enable-Text-Thinking",
        "Enable model thinking for Phase 1 semantic descriptions (default false; set true to enable)",
    ),
)

VISION_TEXT_HEADERS_DOCS = _headers_docs(
    ("X-Api-Key", "API key"),
    ("X-Base-Url", "Chat completions URL"),
    ("X-Model", "Vision model (image analysis)"),
    ("X-Text-Model", "Text model (text analysis)"),
)


# Common Header() kwargs reused across deps
_API_KEY = dict(
    alias="X-Api-Key",
    description="OpenRouter API key. Overrides `OPENROUTER_API_KEY`. Alias: `X-OpenRouter-Api-Key`.",
    example="sk-or-v1-...",
)
_BASE_URL = dict(
    alias="X-Base-Url",
    description="Chat completions URL. Overrides `OPENROUTER_BASE_URL`. Alias: `X-OpenRouter-Base-Url`.",
    example="https://openrouter.ai/api/v1/chat/completions",
)
_EMBEDDING_URL = dict(
    alias="X-Embedding-Url",
    description="Embeddings URL. Overrides `OPENROUTER_EMBEDDING_URL`. Alias: `X-OpenRouter-Embedding-Url`.",
    example="https://openrouter.ai/api/v1/embeddings",
)
_MODEL = dict(
    alias="X-Model",
    description="Vision model name. Overrides the default vision model. Alias: `X-Model-Name`.",
    example="qwen/qwen3-vl-235b-a22b-instruct",
)
_TEXT_MODEL = dict(
    alias="X-Text-Model",
    description="Text model name. Overrides the default text model.",
    example="qwen/qwen3-235b-a22b",
)
_TEXT_MODEL_TRANSLATION = dict(
    alias="X-Text-Model",
    description=(
        "Text model name. Overrides the default text model. "
        "Takes precedence over `X-Model` and the body `model` field."
    ),
    example="qwen/qwen3-235b-a22b",
)
_MODEL_TRANSLATION = dict(
    alias="X-Model",
    description=(
        "Text model name. Overrides body `model` when `X-Text-Model` is not set. "
        "Alias: `X-Model-Name`."
    ),
    example="qwen/qwen3-235b-a22b",
)
_EMBEDDING_MODEL = dict(
    alias="X-Embedding-Model",
    description="Embedding model name. Overrides the default embedding model.",
    example="qwen/qwen3-embedding-8b",
)
_ENABLE_TEXT_THINKING = dict(
    alias="X-Enable-Text-Thinking",
    description=(
        "Enable model thinking for text-model calls on this request. "
        "Defaults to false when omitted. Set to true/1/on to enable. "
        "When false, sends OpenRouter reasoning.effort=none."
    ),
    example="false",
)


def get_vision_overrides(
    x_api_key: Optional[str] = Header(None, **_API_KEY),
    x_openrouter_api_key: Optional[str] = Header(
        None, alias="X-OpenRouter-Api-Key", include_in_schema=False
    ),
    x_base_url: Optional[str] = Header(None, **_BASE_URL),
    x_openrouter_base_url: Optional[str] = Header(
        None, alias="X-OpenRouter-Base-Url", include_in_schema=False
    ),
    x_model: Optional[str] = Header(None, **_MODEL),
    x_model_name: Optional[str] = Header(
        None, alias="X-Model-Name", include_in_schema=False
    ),
) -> OpenRouterHeaderOverrides:
    """Headers used by vision-only endpoints."""
    return OpenRouterHeaderOverrides(
        api_key=_first_non_empty(x_api_key, x_openrouter_api_key),
        base_url=_first_non_empty(x_base_url, x_openrouter_base_url),
        model=_first_non_empty(x_model, x_model_name),
    )


def get_listing_description_overrides(
    x_api_key: Optional[str] = Header(None, **_API_KEY),
    x_openrouter_api_key: Optional[str] = Header(
        None, alias="X-OpenRouter-Api-Key", include_in_schema=False
    ),
    x_base_url: Optional[str] = Header(None, **_BASE_URL),
    x_openrouter_base_url: Optional[str] = Header(
        None, alias="X-OpenRouter-Base-Url", include_in_schema=False
    ),
    x_model: Optional[str] = Header(None, **_MODEL),
    x_model_name: Optional[str] = Header(
        None, alias="X-Model-Name", include_in_schema=False
    ),
    x_enable_text_thinking: Optional[str] = Header(None, **_ENABLE_TEXT_THINKING),
) -> OpenRouterHeaderOverrides:
    """Headers used by listing-description (vision + optional text thinking)."""
    return OpenRouterHeaderOverrides(
        api_key=_first_non_empty(x_api_key, x_openrouter_api_key),
        base_url=_first_non_empty(x_base_url, x_openrouter_base_url),
        model=_first_non_empty(x_model, x_model_name),
        enable_text_thinking=_parse_bool_header(x_enable_text_thinking, default=False),
    )


def get_text_overrides(
    x_api_key: Optional[str] = Header(None, **_API_KEY),
    x_openrouter_api_key: Optional[str] = Header(
        None, alias="X-OpenRouter-Api-Key", include_in_schema=False
    ),
    x_base_url: Optional[str] = Header(None, **_BASE_URL),
    x_openrouter_base_url: Optional[str] = Header(
        None, alias="X-OpenRouter-Base-Url", include_in_schema=False
    ),
    x_model: Optional[str] = Header(None, **_MODEL_TRANSLATION),
    x_model_name: Optional[str] = Header(
        None, alias="X-Model-Name", include_in_schema=False
    ),
    x_text_model: Optional[str] = Header(None, **_TEXT_MODEL_TRANSLATION),
    x_enable_text_thinking: Optional[str] = Header(None, **_ENABLE_TEXT_THINKING),
) -> OpenRouterHeaderOverrides:
    """Headers used by text-only endpoints (e.g. translation)."""
    return OpenRouterHeaderOverrides(
        api_key=_first_non_empty(x_api_key, x_openrouter_api_key),
        base_url=_first_non_empty(x_base_url, x_openrouter_base_url),
        model=_first_non_empty(x_model, x_model_name),
        text_model=_first_non_empty(x_text_model),
        enable_text_thinking=_parse_bool_header(x_enable_text_thinking, default=False),
    )


def get_vision_embedding_overrides(
    x_api_key: Optional[str] = Header(None, **_API_KEY),
    x_openrouter_api_key: Optional[str] = Header(
        None, alias="X-OpenRouter-Api-Key", include_in_schema=False
    ),
    x_base_url: Optional[str] = Header(None, **_BASE_URL),
    x_openrouter_base_url: Optional[str] = Header(
        None, alias="X-OpenRouter-Base-Url", include_in_schema=False
    ),
    x_embedding_url: Optional[str] = Header(None, **_EMBEDDING_URL),
    x_openrouter_embedding_url: Optional[str] = Header(
        None, alias="X-OpenRouter-Embedding-Url", include_in_schema=False
    ),
    x_model: Optional[str] = Header(None, **_MODEL),
    x_model_name: Optional[str] = Header(
        None, alias="X-Model-Name", include_in_schema=False
    ),
    x_embedding_model: Optional[str] = Header(None, **_EMBEDDING_MODEL),
) -> OpenRouterHeaderOverrides:
    """Headers used by vision + embedding endpoints (image-to-form semantic search)."""
    return OpenRouterHeaderOverrides(
        api_key=_first_non_empty(x_api_key, x_openrouter_api_key),
        base_url=_first_non_empty(x_base_url, x_openrouter_base_url),
        embedding_url=_first_non_empty(x_embedding_url, x_openrouter_embedding_url),
        model=_first_non_empty(x_model, x_model_name),
        embedding_model=_first_non_empty(x_embedding_model),
    )


def get_text_embedding_overrides(
    x_api_key: Optional[str] = Header(None, **_API_KEY),
    x_openrouter_api_key: Optional[str] = Header(
        None, alias="X-OpenRouter-Api-Key", include_in_schema=False
    ),
    x_base_url: Optional[str] = Header(None, **_BASE_URL),
    x_openrouter_base_url: Optional[str] = Header(
        None, alias="X-OpenRouter-Base-Url", include_in_schema=False
    ),
    x_embedding_url: Optional[str] = Header(None, **_EMBEDDING_URL),
    x_openrouter_embedding_url: Optional[str] = Header(
        None, alias="X-OpenRouter-Embedding-Url", include_in_schema=False
    ),
    x_text_model: Optional[str] = Header(None, **_TEXT_MODEL),
    x_embedding_model: Optional[str] = Header(None, **_EMBEDDING_MODEL),
    x_enable_text_thinking: Optional[str] = Header(None, **_ENABLE_TEXT_THINKING),
) -> OpenRouterHeaderOverrides:
    """Headers used by text + embedding endpoints (category embeddings)."""
    return OpenRouterHeaderOverrides(
        api_key=_first_non_empty(x_api_key, x_openrouter_api_key),
        base_url=_first_non_empty(x_base_url, x_openrouter_base_url),
        embedding_url=_first_non_empty(x_embedding_url, x_openrouter_embedding_url),
        text_model=_first_non_empty(x_text_model),
        embedding_model=_first_non_empty(x_embedding_model),
        enable_text_thinking=_parse_bool_header(x_enable_text_thinking, default=False),
    )


def get_vision_text_overrides(
    x_api_key: Optional[str] = Header(None, **_API_KEY),
    x_openrouter_api_key: Optional[str] = Header(
        None, alias="X-OpenRouter-Api-Key", include_in_schema=False
    ),
    x_base_url: Optional[str] = Header(None, **_BASE_URL),
    x_openrouter_base_url: Optional[str] = Header(
        None, alias="X-OpenRouter-Base-Url", include_in_schema=False
    ),
    x_model: Optional[str] = Header(None, **_MODEL),
    x_model_name: Optional[str] = Header(
        None, alias="X-Model-Name", include_in_schema=False
    ),
    x_text_model: Optional[str] = Header(None, **_TEXT_MODEL),
) -> OpenRouterHeaderOverrides:
    """Headers used by endpoints that call both vision and text models."""
    return OpenRouterHeaderOverrides(
        api_key=_first_non_empty(x_api_key, x_openrouter_api_key),
        base_url=_first_non_empty(x_base_url, x_openrouter_base_url),
        model=_first_non_empty(x_model, x_model_name),
        text_model=_first_non_empty(x_text_model),
    )


# Backward-compatible alias used if any caller still imports the full set
get_openrouter_overrides = get_vision_embedding_overrides


def create_openrouter_service(
    overrides: Optional[OpenRouterHeaderOverrides] = None,
) -> OpenRouterService:
    """Create an OpenRouterService, applying header overrides when present."""
    overrides = overrides or OpenRouterHeaderOverrides()
    return OpenRouterService(
        api_key=overrides.api_key,
        base_url=overrides.base_url,
        embedding_url=overrides.embedding_url,
        model=overrides.model,
        text_model=overrides.text_model,
        embedding_model=overrides.embedding_model,
        enable_text_thinking=overrides.enable_text_thinking,
    )
