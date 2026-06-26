from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class EmbeddingRecord(BaseModel):
    """Registro de la base vectorial (estructura sugerida por el TP)."""

    id_imagen: str
    embedding: list[float]
    path: str
    breed: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Neighbor(BaseModel):
    """Un vecino recuperado por la busqueda por similitud."""

    path: str
    breed: str
    score: float
    url: Optional[str] = Field(
        default=None,
        description="URL relativa al API para descargar la imagen (/files/data/...).",
    )


class SearchRequest(BaseModel):
    source_path: str
    model: Optional[str] = Field(
        default=None,
        description="baseline | resnet18_finetuned | cnn_custom (default: EMBEDDING_MODEL).",
    )
    top_k: Optional[int] = Field(default=None, description="Cantidad de vecinos (default: TOP_K).")


class SearchResult(BaseModel):
    """Resultado de la Etapa 1: imagen consultada, top K similares y raza predicha."""

    type: Literal["search"] = "search"
    source_path: str
    model: str
    predicted_breed: str
    score: float
    neighbors: list[Neighbor]


class ClassifyRequest(BaseModel):
    source_path: str
    model: Optional[str] = Field(
        default=None,
        description="resnet18_finetuned | cnn_custom (default: resnet18_finetuned).",
    )


class ClassifyResult(BaseModel):
    """Resultado de la Etapa 2: raza predicha por el clasificador entrenado."""

    type: Literal["classify"] = "classify"
    source_path: str
    model: str
    breed: str
    score: float
    gradcam_path: Optional[str] = None
    gradcam_url: Optional[str] = None
    ood_detected: Optional[bool] = None
    ood_score: Optional[float] = None
    top_5: Optional[list[dict[str, Any]]] = None


class DetectRequest(BaseModel):
    source_path: str


class DogDetection(BaseModel):
    """Una deteccion del pipeline (Etapa 3)."""

    bbox: list[int]
    det_score: float
    breed: str
    breed_score: float
    gradcam_path: Optional[str] = None
    gradcam_url: Optional[str] = None
    top_5: Optional[list[dict[str, Any]]] = None


class DetectResult(BaseModel):
    """Resultado de la Etapa 3: bounding boxes, razas y scores de confianza."""

    type: Literal["detect"] = "detect"
    source_path: str
    detections: list[DogDetection]
    detected_breeds: list[str]
    ood_detected: Optional[bool] = None
    ood_score: Optional[float] = None


class AsyncTaskCreated(BaseModel):
    status: Literal["accepted"] = "accepted"
    job_id: str


class UploadResponse(BaseModel):
    """Respuesta tras subir un archivo al servidor (rutas usadas por /search y /detect)."""

    path: str
    download_url: str


class StatusResponse(BaseModel):
    status: Literal["done", "inProgress", "failed"]
    link: str
    reason: Optional[str] = None
    artifact_url: Optional[str] = Field(
        default=None,
        description="URL relativa al API del artefacto principal (.json del resultado).",
    )
    source_image_url: Optional[str] = Field(
        default=None,
        description="URL relativa de la imagen origen (consulta o imagen procesada).",
    )


class ModelsResponse(BaseModel):
    models: list[str]
    selected: str
