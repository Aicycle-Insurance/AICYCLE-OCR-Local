import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", PROJECT_ROOT / "config.yaml"))

# ${ENV_NAME} hoặc ${ENV_NAME:default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


class AppConfig(BaseModel):
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
    log_level: str = "INFO"


class OCREngineConfig(BaseModel):
    base_url: str
    api_key: str = "EMPTY"
    model: str
    timeout: float = 600
    temperature: float = 0.0
    top_p: Optional[float] = None
    max_tokens: int = 8192


class RetryConfig(BaseModel):
    max_attempts: int = 3
    initial_delay: float = 2.0
    max_delay: float = 30.0


class OCRConfig(BaseModel):
    engine: str
    dpi: int = 200
    max_concurrency: int = 8
    retry: RetryConfig = RetryConfig()
    engines: dict[str, OCREngineConfig]


class LLMConfig(BaseModel):
    base_url: str
    api_key: str = "EMPTY"
    model: str
    timeout: float = 600
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    max_concurrent_extractions: int = 3
    table_chunk_pages: int = 2
    extra_body: dict[str, Any] = {}


class LangfuseConfig(BaseModel):
    enabled: bool = False
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    base_url: Optional[str] = None


class Settings(BaseModel):
    app: AppConfig
    ocr: OCRConfig
    llm: LLMConfig
    langfuse: LangfuseConfig


def _expand_env(raw: str) -> str:
    """Thay ${ENV:default} bằng giá trị env, không có env thì dùng default."""
    return _ENV_PATTERN.sub(
        lambda m: os.environ.get(m.group(1), m.group(2) or ""), raw
    )


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    data = yaml.safe_load(_expand_env(path.read_text(encoding="utf-8")))
    settings = Settings.model_validate(data)
    for attr in ("data_dir", "log_dir"):
        value = getattr(settings.app, attr)
        if not value.is_absolute():
            setattr(settings.app, attr, PROJECT_ROOT / value)
    return settings


settings = load_settings()
