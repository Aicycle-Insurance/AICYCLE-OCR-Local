import sys

from loguru import logger

from src.config import settings


def setup_logger(service_name: str) -> None:
    """Log ra console và file logs/{service_name}.log."""
    logger.remove()
    logger.add(sys.stderr, level=settings.app.log_level)
    settings.app.log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.app.log_dir / f"{service_name}.log",
        level=settings.app.log_level,
        rotation="50 MB",
        retention=10,
        enqueue=True,
        colorize=False,
        encoding="utf-8",
    )
