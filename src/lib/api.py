from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse

from lib.bootstrap import build_services
from lib.config import settings
from lib.files import file_to_public_url, safe_file_under
from lib.schemas import (
    AsyncTaskCreated,
    ClassifyRequest,
    DetectRequest,
    ModelsResponse,
    SearchRequest,
    StatusResponse,
    UploadResponse,
)
from lib.services.task_manager import TaskManager
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dog-recognition"])
task_manager = TaskManager()

PUBLIC_ROOTS = (
    (settings.output_path, "/files/output"),
    (settings.data_path, "/files/data"),
)


def _public_url(path: Path) -> str | None:
    return file_to_public_url(path, PUBLIC_ROOTS)


services = build_services(settings, url_resolver=_public_url)
similarity_service = services.similarity
classifier_service = services.classifier
detection_service = services.detection

EMBEDDING_MODELS = ("baseline",)
CLASSIFIER_MODELS = ("resnet18_finetuned", "cnn_custom")


def _embedding_extractor(model_name: str):
    if model_name not in EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Expected one of: {EMBEDDING_MODELS}",
        )
    return similarity_service.extract_embedding


def _urls_for_status(link: str) -> tuple[str | None, str | None]:
    if link in ("", "none"):
        return None, None
    p = Path(link)
    if not p.is_file():
        return None, None
    artifact = _public_url(p)
    source: str | None = None
    if p.suffix.lower() == ".json":
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            sp = payload.get("source_path")
            if isinstance(sp, str) and sp:
                source = _public_url(Path(sp))
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    return artifact, source


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    uploads_dir = settings.output_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    raw_name = (file.filename or "upload").strip()
    suffix = Path(raw_name).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
        suffix = ".jpg"
    dest = uploads_dir / f"up_{uuid4().hex}{suffix}"
    data = await file.read()
    dest.write_bytes(data)
    rel = dest.resolve().relative_to(settings.output_path.resolve())
    return UploadResponse(
        path=str(dest.resolve()),
        download_url=f"/files/output/{rel.as_posix()}",
    )


@router.get("/files/output/{file_path:path}")
async def download_output_file(file_path: str) -> FileResponse:
    path = safe_file_under(settings.output_path, file_path)
    media, _ = mimetypes.guess_type(path.name)
    return FileResponse(path, filename=path.name, media_type=media or "application/octet-stream")


@router.get("/files/data/{file_path:path}")
async def download_data_file(file_path: str) -> FileResponse:
    path = safe_file_under(settings.data_path, file_path)
    media, _ = mimetypes.guess_type(path.name)
    return FileResponse(path, filename=path.name, media_type=media or "application/octet-stream")


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """Modelos disponibles para generar embeddings en la busqueda por similitud."""
    return ModelsResponse(models=list(EMBEDDING_MODELS), selected=settings.embedding_model)


@router.post("/search", response_model=AsyncTaskCreated)
async def search(payload: SearchRequest, response: Response) -> AsyncTaskCreated:
    """Etapa 1: busqueda por similitud (imagen consultada, top K similares, raza predicha)."""
    response.status_code = status.HTTP_202_ACCEPTED
    model_name = payload.model or settings.embedding_model
    extractor = _embedding_extractor(model_name)
    job_id = task_manager.create_job()

    async def _process() -> str:
        await asyncio.sleep(0.05)
        logger.info("Searching similar images for: %s (model=%s)", payload.source_path, model_name)
        return similarity_service.search(
            source_path=payload.source_path,
            output_path=settings.output_path,
            embedding_fn=extractor,
            model_name=model_name,
            top_k=payload.top_k,
        )

    task_manager.schedule(job_id, _process())
    return AsyncTaskCreated(job_id=job_id)


@router.post("/classify", response_model=AsyncTaskCreated)
async def classify(payload: ClassifyRequest, response: Response) -> AsyncTaskCreated:
    """Etapa 2: clasificacion supervisada de una imagen con el modelo entrenado."""
    response.status_code = status.HTTP_202_ACCEPTED
    model_name = payload.model or "resnet18_finetuned"
    if model_name not in CLASSIFIER_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Expected one of: {CLASSIFIER_MODELS}",
        )
    job_id = task_manager.create_job()

    async def _process() -> str:
        await asyncio.sleep(0.05)
        logger.info("Classifying image: %s (model=%s)", payload.source_path, model_name)
        return detection_service.classify_image(
            payload.source_path, settings.output_path, model_name
        )

    task_manager.schedule(job_id, _process())
    return AsyncTaskCreated(job_id=job_id)


@router.post("/detect", response_model=AsyncTaskCreated)
async def detect(payload: DetectRequest, response: Response) -> AsyncTaskCreated:
    """Etapa 3: deteccion de perros + clasificacion de cada recorte."""
    response.status_code = status.HTTP_202_ACCEPTED
    job_id = task_manager.create_job()

    async def _process() -> str:
        await asyncio.sleep(0.05)
        logger.info("Detecting dogs in: %s", payload.source_path)
        return detection_service.predict(payload.source_path, settings.output_path)

    task_manager.schedule(job_id, _process())
    return AsyncTaskCreated(job_id=job_id)


@router.get("/status/{job_id}", response_model=StatusResponse)
async def status_by_id(job_id: str) -> StatusResponse:
    state = task_manager.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    artifact_url: str | None = None
    source_image_url: str | None = None
    if state.status == "done":
        artifact_url, source_image_url = _urls_for_status(state.link)
    return StatusResponse(
        status=state.status,
        link=state.link,
        reason=state.error,
        artifact_url=artifact_url,
        source_image_url=source_image_url,
    )
