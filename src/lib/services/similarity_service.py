from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np

from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)


class SimilarityService:
    """Etapa 1: buscador de imagenes por similitud.

    Funciones a implementar por el estudiante:
      - extract_embedding(image)
      - search_similar_images(embedding, top_k)
      - predict_breed_from_neighbors(results)

    La orquestacion (search, index_image, persistencia y metricas de similitud)
    ya esta provista y no debe modificarse sin justificarlo en el informe.
    """

    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        top_k: int,
        image_size: int,
        model_name: str,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.image_size = image_size
        self.model_name = model_name
        self.url_resolver = url_resolver

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV)
        return image

    # ------------------------------------------------------------------
    # Etapa 1: funciones a implementar
    # ------------------------------------------------------------------

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando un modelo pre-entrenado en
        ImageNet (ej: ResNet50, EfficientNet, ConvNeXt) sin la capa de
        clasificacion final.
        """
        import torch
        from torchvision import models, transforms

        # Cargamos el modelo la primera vez que se ejecuta y lo guardamos en el objeto (lazy loading).
        if not hasattr(self, "_baseline_model"):
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)
            # Reemplazamos la capa final por una capa Lineal para proyectar de 2048 a 512 dimensiones
            model.fc = torch.nn.Linear(2048, 512)
            model.eval()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._baseline_model = model.to(self._device)

        # 1. Convertir la imagen de BGR (OpenCV) a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Definir las transformaciones incluyendo la normalizacion estandar de ImageNet
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Aplicamos el preprocesamiento y añadimos la dimension del batch (B, C, H, W)
        input_tensor = preprocess(image_rgb).unsqueeze(0).to(self._device)

        # 3. Extraer el embedding
        with torch.no_grad():
            embedding = self._baseline_model(input_tensor)
            # Normalización L2 para asegurar que la norma sea 1, 
            # haciendo que la similitud del coseno sea equivalente al producto punto
            import torch.nn.functional as F
            embedding = F.normalize(embedding, p=2, dim=1)

        # 4. Convertir el tensor a una lista de floats y retornar
        return embedding.squeeze().cpu().tolist()

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """
        Recupera de la base vectorial las top_k imagenes mas similares.
        """
        records = self.store.search(embedding, k=top_k)
        neighbors = []
        for r in records:
            score = self.similarity(embedding, r.embedding)
            neighbors.append(Neighbor(path=r.path, breed=r.breed, score=score))
        
        neighbors.sort(key=lambda x: x.score, reverse=True)
        return neighbors

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Predice la raza a partir de los vecinos recuperados mediante voto
        mayoritario ponderado por score.
        """
        if not results:
            return "unknown", 0.0

        # Los resultados ya vienen ordenados, el primero tiene el mejor score
        if results[0].score < self.similarity_threshold:
            return "unknown", results[0].score

        # Sumamos los scores de cada raza (voto ponderado)
        breed_votes = {}
        for n in results:
            breed_votes[n.breed] = breed_votes.get(n.breed, 0.0) + n.score

        # Elegimos la raza con la mayor suma acumulada
        best_breed = max(breed_votes.items(), key=lambda x: x[1])[0]

        # Retornamos la raza ganadora y el score del vecino mas cercano de esa raza
        best_score_for_breed = next(n.score for n in results if n.breed == best_breed)
        return best_breed, best_score_for_breed

    # ------------------------------------------------------------------
    # Helpers de similitud provistos
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def index_image(
        self, image_path: str, breed: str, metadata: dict[str, object] | None = None
    ) -> EmbeddingRecord:
        """Extrae el embedding de una imagen del dataset y lo persiste en la base vectorial."""
        image = self._load_image(image_path)
        embedding = self.extract_embedding(image)
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(image_path),
            breed=breed,
            metadata=metadata or {},
        )
        self.store.append(record)
        return record

    def _with_url(self, neighbor: Neighbor) -> Neighbor:
        if self.url_resolver is not None and not neighbor.url:
            neighbor.url = self.url_resolver(Path(neighbor.path))
        return neighbor

    def search(
        self,
        source_path: str,
        output_path: Path,
        embedding_fn: Optional[Callable[[np.ndarray], list[float]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Pipeline completo de la Etapa 1: embedding -> vecinos -> raza predicha.

        `embedding_fn` permite seleccionar dinamicamente el extractor
        (baseline, resnet18_finetuned o cnn_custom, ver Etapa 2).
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        extractor = embedding_fn or self.extract_embedding
        embedding = extractor(image)

        k = int(top_k) if top_k else self.top_k
        neighbors = [self._with_url(n) for n in self.search_similar_images(embedding, k)]
        breed, score = self.predict_breed_from_neighbors(neighbors)
        logger.info("Predicted breed: %s (score=%.4f) for %s", breed, score, source_path)

        payload = SearchResult(
            source_path=source_path,
            model=model_name or self.model_name,
            predicted_breed=breed,
            score=round(float(score), 4),
            neighbors=neighbors,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
