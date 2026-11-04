"""Content-addressed stage cache under data/cache/stages/ (plan 1.4)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import orjson

CODE_VERSION = "0.1"


def stage_key(stage: str, inputs: Any, params: Any = None, code_version: str = CODE_VERSION) -> str:
    """sha256 over (stage, inputs, params, code_version); inputs/params must be JSON-able
    (pass file hashes rather than file contents)."""
    blob = orjson.dumps(
        {"stage": stage, "inputs": inputs, "params": params, "code": code_version},
        option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS,
        default=str,
    )
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


class StageCache:
    """Each key owns a directory `<root>/<key[:2]>/<key>/`; a `.done` marker makes writes atomic-ish."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def dir_for(self, key: str) -> Path:
        return self.root / key[:2] / key

    def has(self, key: str) -> bool:
        return (self.dir_for(key) / ".done").exists()

    def begin(self, key: str) -> Path:
        d = self.dir_for(key)
        d.mkdir(parents=True, exist_ok=True)
        (d / ".done").unlink(missing_ok=True)
        return d

    def commit(self, key: str) -> Path:
        d = self.dir_for(key)
        (d / ".done").write_text("")
        return d

    def get_dir(self, key: str) -> Path | None:
        return self.dir_for(key) if self.has(key) else None

    def put_json(self, key: str, obj: Any, name: str = "value.json") -> Path:
        d = self.begin(key)
        (d / name).write_text(json.dumps(obj, default=str))
        self.commit(key)
        return d / name

    def get_json(self, key: str, name: str = "value.json") -> Any | None:
        d = self.get_dir(key)
        if d is None or not (d / name).exists():
            return None
        return json.loads((d / name).read_text())
