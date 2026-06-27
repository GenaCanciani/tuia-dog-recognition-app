from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fastapi import HTTPException


def safe_file_under(root: Path, relpath: str) -> Path:
    """Resuelve `relpath` dentro de `root` evitando path traversal."""
    root = root.resolve()
    rel = (relpath or "").replace("\\", "/").strip("/")
    if not rel or ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="invalid path")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path outside allowed root") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return candidate


def file_to_public_url(path: Path, roots: Iterable[tuple[Path, str]]) -> str | None:
    """Mapea un path local a una URL publica si cae bajo alguno de los roots servidos.

    Incluye un fallback por sufijo para paths indexados en un entorno distinto
    al que corre el servidor (ej: indexado local, servidor en Docker).
    """
    try:
        p = Path(path).resolve()
    except OSError:
        p = None
    # Intento estricto: relative_to con paths resueltos
    if p is not None:
        for base, prefix in roots:
            try:
                rel = p.relative_to(Path(base).resolve())
                return f"{prefix}/{rel.as_posix()}"
            except ValueError:
                continue
    # Fallback: buscar el nombre del directorio raiz como componente en el path
    path_posix = Path(path).as_posix()
    for base, prefix in roots:
        base_name = Path(base).name  # ej: "data", "output"
        marker = f"/{base_name}/"
        idx = path_posix.find(marker)
        if idx != -1:
            rel = path_posix[idx + len(marker):]
            return f"{prefix}/{rel}"
    return None
