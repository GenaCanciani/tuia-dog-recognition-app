from __future__ import annotations

import json
from pathlib import Path

from lib.schemas import EmbeddingRecord


class EmbeddingStore:
    """Base vectorial simple sobre un archivo JSON (alternativa a pgvector)."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text("[]", encoding="utf-8")

    def all(self) -> list[EmbeddingRecord]:
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        return [EmbeddingRecord.model_validate(item) for item in payload]

    def save(self, records: list[EmbeddingRecord]) -> None:
        raw = [item.model_dump() for item in records]
        self.storage_path.write_text(
            json.dumps(raw, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def append(self, record: EmbeddingRecord) -> None:
        records = self.all()
        records.append(record)
        self.save(records)