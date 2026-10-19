"""Langfuse tracing (fail-open), port từ Gemini-OCR."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any, Iterator

from loguru import logger

from src.config import settings

_trace_scope_active: ContextVar[bool] = ContextVar(
    "langfuse_trace_scope_active", default=False
)
_client: Any = None
_client_initialized = False
_client_lock = Lock()


class NoOpObservation:
    """Observation-compatible object used whenever tracing is unavailable."""

    def update(self, **_: Any) -> "NoOpObservation":
        return self


def _get_client() -> Any | None:
    """Create the Langfuse client once; any failure permanently disables it."""
    global _client, _client_initialized

    if _client_initialized:
        return _client

    with _client_lock:
        if _client_initialized:
            return _client
        _client_initialized = True

        config = settings.langfuse
        if not config.enabled:
            logger.info("Langfuse tracing is disabled")
            return None

        required = {
            "LANGFUSE_PUBLIC_KEY": config.public_key,
            "LANGFUSE_SECRET_KEY": config.secret_key,
            "LANGFUSE_BASE_URL": config.base_url,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            logger.warning(
                "Langfuse tracing disabled because configuration is incomplete: {}",
                ", ".join(missing),
            )
            return None

        try:
            from langfuse import Langfuse

            _client = Langfuse(
                public_key=config.public_key,
                secret_key=config.secret_key,
                base_url=config.base_url,
            )
            logger.info("Langfuse tracing initialized")
        except Exception:
            _client = None
            logger.exception("Failed to initialize Langfuse; service will continue")

        return _client


@contextmanager
def _safe_observation(
    *,
    name: str,
    as_type: str = "span",
    **kwargs: Any,
) -> Iterator[Any]:
    """Create an observation without allowing tracing failures into app logic."""
    client = _get_client()
    if client is None:
        yield NoOpObservation()
        return

    manager = None
    observation: Any = NoOpObservation()
    try:
        manager = client.start_as_current_observation(
            name=name,
            as_type=as_type,
            **kwargs,
        )
        observation = manager.__enter__()
    except Exception:
        logger.exception("Failed to start Langfuse observation: {}", name)
        yield observation
        return

    try:
        yield observation
    except BaseException as application_error:
        try:
            manager.__exit__(
                type(application_error), application_error, application_error.__traceback__
            )
        except Exception:
            logger.exception("Failed to close Langfuse observation: {}", name)
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception:
            logger.exception("Failed to close Langfuse observation: {}", name)


@contextmanager
def trace_pipeline(name: str, **kwargs: Any) -> Iterator[Any]:
    """Start a root trace scope for one request (OCR run or extraction)."""
    token = _trace_scope_active.set(True)
    try:
        with _safe_observation(name=name, as_type="span", **kwargs) as observation:
            yield observation
    finally:
        _trace_scope_active.reset(token)


@contextmanager
def trace_observation(
    name: str,
    *,
    as_type: str = "span",
    **kwargs: Any,
) -> Iterator[Any]:
    """Create a nested observation only inside an explicitly traced pipeline."""
    if not _trace_scope_active.get():
        yield NoOpObservation()
        return

    with _safe_observation(name=name, as_type=as_type, **kwargs) as observation:
        yield observation


def update_observation(observation: Any, **kwargs: Any) -> None:
    """Update an observation while preserving fail-open tracing semantics."""
    try:
        observation.update(**kwargs)
    except Exception:
        logger.exception("Failed to update Langfuse observation; service will continue")


def shutdown_langfuse() -> None:
    """Flush and stop the client without affecting application shutdown."""
    global _client, _client_initialized

    client = _client
    if client is None:
        return
    try:
        client.shutdown()
    except Exception:
        logger.exception("Failed to shut down Langfuse cleanly")
    finally:
        _client = None
        _client_initialized = False
