"""Phần dùng chung cho các script Label Studio (LS): session, task, map output -> prediction.

Logic map prediction giữ nguyên như test/ocr_labelstudio_runner/run.py để
predictions_preview.json dùng được với evaluate.py cũ.
"""

import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests


def get_api_key() -> str:
    # src.config đã load .env vào os.environ
    key = os.environ.get("LABEL_STUDIO_API_KEY")
    if not key:
        raise SystemExit("Set LABEL_STUDIO_API_KEY (Personal Access Token) in .env")
    return key


def field_name(code, key):
    return f"{code}__{key}"


class LSSession(requests.Session):
    """Session tự lấy lại access token khi gặp 401 (token ngắn hạn, chạy lâu sẽ hết hạn)."""

    def __init__(self, base, refresh):
        super().__init__()
        self.base, self.refresh = base, refresh
        self._login()

    def _login(self):
        r = requests.post(f"{self.base}/api/token/refresh", json={"refresh": self.refresh}, timeout=30)
        r.raise_for_status()
        self.headers["Authorization"] = f"Bearer {r.json()['access']}"

    def request(self, method, url, **kw):
        r = super().request(method, url, **kw)
        if r.status_code == 401:
            self._login()
            r = super().request(method, url, **kw)
        return r


def api_get(session, base, route, params=None):
    r = session.get(f"{base}{route}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def list_tasks(session, base, project):
    # SDK/server versions differ in pagination. Handle both list and paginated responses.
    page = 1
    all_tasks = []
    while True:
        result = api_get(
            session,
            base,
            "/api/tasks/",
            {"project": project, "page": page, "page_size": 100},
        )
        if isinstance(result, list):
            all_tasks.extend(result)
            break
        if (
            not isinstance(result, dict)
            or "tasks" not in result
            and "results" not in result
        ):
            raise ValueError(f"Unexpected tasks response: {type(result)}")
        batch = result.get("tasks", result.get("results", []))
        all_tasks.extend(batch)
        if not result.get("next"):
            break
        page += 1
    return all_tasks


def get_tasks(session, base, project, task_id=None):
    if task_id:
        return [api_get(session, base, f"/api/tasks/{task_id}/")]
    return list_tasks(session, base, project)


def download_task_file(session, base, task) -> tuple[bytes, str]:
    """Tải file của task (data.file_url hoặc data.image), trả về (bytes, file name)."""
    data = task.get("data") or {}
    url = data.get("file_url") or data.get("image")
    if not isinstance(url, str) or not url:
        raise ValueError("no file_url/image in task data")

    parsed = urlparse(url)
    if url.startswith("/") or url.startswith(base):
        # File host trên LS (/data/upload/..., /data/local-files/?d=...) cần auth
        r = session.get(url if url.startswith(base) else f"{base}{url}", timeout=120)
    elif parsed.scheme in ("http", "https"):
        r = requests.get(url, timeout=120)
    else:
        raise ValueError(f"unsupported file url: {url}")
    r.raise_for_status()

    # /data/local-files/?d=dir/file.pdf -> lấy tên file từ tham số d
    local_path = parse_qs(parsed.query).get("d", [""])[0]
    name = Path(local_path or parsed.path).name or f"task_{task['id']}"
    return r.content, name


def unbox(v):
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def fmt_cell(v):
    """Một ô của bảng, giống fill.py: null -> rỗng, ô có dấu phẩy/nháy được bọc "..."."""
    v = unbox(v)
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    s = str(v).replace("\n", " ").strip()
    if any(c in s for c in ',"'):
        s = '"' + s.replace('"', '""') + '"'
    return s


def row_text(row, child_schema):
    """Một dòng của bảng -> "ô1, ô2, ..." theo thứ tự child_schema (định dạng ground truth của fill.py)."""
    if isinstance(row, dict):
        keys = [c["field_key"] for c in child_schema] if child_schema else list(row)
        cells = [row.get(k) for k in keys]
    else:
        cells = list(row)
    return ", ".join(fmt_cell(c) for c in cells)


def make_result(code, values, schema):
    fields = {f["field_key"]: f["data_type"] for f in schema["fields"]}
    child_schemas = {f["field_key"]: f.get("child_schema") for f in schema["fields"]}
    output = [
        {
            "from_name": "schema_type",
            "to_name": "document",
            "type": "choices",
            "value": {"choices": [code]},
        }
    ]
    for key, kind in fields.items():
        v = unbox(values.get(key))
        name = field_name(code, key)
        if kind == "boolean":
            if v is None:
                choice = "null"
            elif isinstance(v, bool):
                choice = str(v).lower()
            elif isinstance(v, str) and v.lower() in ("true", "false", "null"):
                choice = v.lower()
            else:
                raise ValueError(f"Invalid boolean {key}={v!r}")
            output.append(
                {
                    "from_name": name,
                    "to_name": "document",
                    "type": "choices",
                    "value": {"choices": [choice]},
                }
            )
        elif kind == "array":
            if v is None:
                continue
            if not isinstance(v, list):
                raise ValueError(f"Invalid array {key}: {type(v).__name__}")
            # Mỗi dòng là 1 phần tử của text, giống ground truth do fill.py / bảng trên LS ghi
            rows = [row_text(r, child_schemas[key]) for r in v]
            rows = [r for r in rows if r]
            if rows:
                output.append(
                    {
                        "from_name": name,
                        "to_name": "document",
                        "type": "textarea",
                        "value": {"text": rows},
                    }
                )
        elif v is not None:
            output.append(
                {
                    "from_name": name,
                    "to_name": "document",
                    "type": "textarea",
                    "value": {"text": [str(v)]},
                }
            )
    return output


def extract_one(response, schema_map):
    docs = response.get("documents", [])
    if len(docs) != 1:
        raise ValueError(f"Expected exactly 1 document, got {len(docs)}")
    doc = docs[0]
    code = doc.get("document_code")
    if code not in schema_map:
        raise ValueError(f"Unrecognized document_code: {code!r}")
    extracted = doc.get("extracted_data")
    if (
        not isinstance(extracted, list)
        or len(extracted) != 1
        or not isinstance(extracted[0], dict)
    ):
        raise ValueError("Expected extracted_data with exactly one object")
    return code, make_result(code, extracted[0], schema_map[code])


def read_runs(path: Path) -> dict[int, tuple[str, str]]:
    """Đọc file runs: mỗi dòng ``task_id<TAB>run_id<TAB>status``."""
    runs = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                task_id, run_id, status = line.split("\t")
                runs[int(task_id)] = (run_id, status)
    return runs


def write_runs(path: Path, runs: dict[int, tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# task_id\tocr_run_id\tstatus"]
    lines += [f"{tid}\t{run_id}\t{status}" for tid, (run_id, status) in sorted(runs.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
