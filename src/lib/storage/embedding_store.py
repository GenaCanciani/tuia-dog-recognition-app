from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lib.schemas import EmbeddingRecord


class EmbeddingStore:
    """Base vectorial simple sobre un archivo JSON (alternativa a pgvector)."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path.resolve()
        self._last_mtime: float = 0
        self._records: list[EmbeddingRecord] = []
        self._reload()

    def _reload(self) -> None:
        if self.storage_path.exists():
            mtime = self.storage_path.stat().st_mtime
            if mtime != self._last_mtime:
                payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
                self._records = [EmbeddingRecord.model_validate(item) for item in payload]
                self._last_mtime = mtime

    def _persist(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.storage_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps([r.model_dump() for r in self._records], ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.storage_path)
        self._last_mtime = self.storage_path.stat().st_mtime

    def all(self) -> list[EmbeddingRecord]:
        self._reload()
        return list(self._records)

    def search(self, query: list[float], k: int = 10, model: str | None = None) -> list[EmbeddingRecord]:
        self._reload()
        query_arr = np.asarray(query, dtype=np.float32)
        scored = []
        for r in self._records:
            if model is not None and r.metadata.get("model") != model:
                continue
            ref_arr = np.asarray(r.embedding, dtype=np.float32)
            sim = float(np.dot(query_arr, ref_arr) / (np.linalg.norm(query_arr) * np.linalg.norm(ref_arr) + 1e-10))
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]

    def delete_by_model(self, model: str) -> None:
        self._records = [r for r in self._records if r.metadata.get("model") != model]

    def append(self, record: EmbeddingRecord) -> None:
        self._records.append(record)

    def flush(self) -> None:
        self._persist()
