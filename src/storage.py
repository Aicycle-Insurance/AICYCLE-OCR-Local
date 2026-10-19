"""Lưu input/output của từng lần chạy vào data/{kind}/{run_id}/ để xem lại."""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import settings


def new_run_dir(kind: str) -> tuple[str, Path]:
    """Tạo thư mục data/{kind}/{run_id}, run_id dạng 20261015_133000_ab12cd34."""
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    run_dir = settings.app.data_dir / kind / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def safe_filename(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^\w.\-]", "_", name) or "input"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
