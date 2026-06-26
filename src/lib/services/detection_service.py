from __future__ import annotations

from typing import Callable, Optional
import json
import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from ultralytics import YOLO

from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService
from lib.services.similarity_service import SimilarityService

logger = logging.getLogger(__name__)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones a implementar por el estudiante:
      - detect_dogs(image)
      - classify_detected_dog(crop)

    La orquestacion (predict: deteccion -> recorte -> clasificacion -> JSON)
    ya esta provista.
    """

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
        similarity: Optional[SimilarityService] = None,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id
        self.url_resolver = url_resolver
        self.similarity = similarity
        self._yolo: YOLO | None = None
        self._class_names: list[str] | None = None

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        return image

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        if self._yolo is None:
            self._yolo = YOLO(self.yolo_model_name)
        results = self._yolo(image, conf=self.conf_threshold, verbose=False)[0]
        detections: list[tuple[tuple[int, int, int, int], float]] = []
        if results.boxes is not None:
            for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                if int(cls_id) == self.dog_class_id:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    detections.append(((x1, y1, x2, y2), float(conf)))
        return detections

    def detect_ood(self, image: np.ndarray) -> tuple[bool, float]:
        """Determina si la imagen es Fuera de Distribución (no perro) usando embeddings de la Etapa 1."""
        if self.similarity is None:
            return False, 0.0
        try:
            emb = self.similarity.extract_embedding(image)
            neighbors = self.similarity.search_similar_images(emb, top_k=5)
            if not neighbors:
                return True, 1.0
            avg_sim = sum(n.score for n in neighbors) / len(neighbors)
            is_ood = avg_sim < 0.65
            ood_score = max(0.0, min(1.0, 1.0 - avg_sim))
            return is_ood, ood_score
        except Exception as e:
            logger.error("Error al detectar OOD: %s", e)
            return False, 0.0

    def classify_detected_dog_with_probs(self, crop: np.ndarray) -> tuple[str, float, list[dict]]:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if self._class_names is None:
            ds = datasets.ImageFolder(str(self.classifier.dataset_path / "train"))
            self._class_names = ds.classes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.classifier.load_model()
        if isinstance(model, nn.Module):
            model = model.to(device)
            model.eval()
        else:
            raise ValueError("classify_detected_dog no soporta ONNX")
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.classifier.image_size, self.classifier.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(crop_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, min(5, len(probs)))
            
            top_5 = []
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                top_5.append({
                    "breed": self._class_names[idx],
                    "score": round(float(prob), 4)
                })
            top_1_breed = top_5[0]["breed"]
            top_1_score = top_5[0]["score"]
        return top_1_breed, top_1_score, top_5

    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        breed, score, _ = self.classify_detected_dog_with_probs(crop)
        return breed, score

    def _generate_gradcam(
        self, image: np.ndarray, breed: str, output_path: Path
    ) -> tuple[Optional[str], Optional[str]]:
        gradcam_path = None
        gradcam_url = None
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = self.classifier.load_model()
            if isinstance(model, nn.Module):
                from lib.visualization.gradcam import GradCAM, get_target_layer, apply_gradcam_overlay
                target_layer = get_target_layer(model, self.classifier.active_model_name)
                if target_layer is not None:
                    if self._class_names is None:
                        ds = datasets.ImageFolder(str(self.classifier.dataset_path / "train"))
                        self._class_names = ds.classes
                    class_idx = self._class_names.index(breed)

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((self.classifier.image_size, self.classifier.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

                    model = model.to(device)
                    model.eval()
                    with torch.enable_grad():
                        gcam = GradCAM(model, target_layer)
                        cam = gcam.generate(input_tensor, class_idx)
                        if cam is not None:
                            overlay = apply_gradcam_overlay(image, cam)
                            gc_dir = output_path / "gradcam"
                            gc_dir.mkdir(parents=True, exist_ok=True)
                            gc_file = gc_dir / f"gc_{uuid4().hex}.jpg"
                            cv2.imwrite(str(gc_file), overlay)
                            gradcam_path = str(gc_file)
                            if self.url_resolver is not None:
                                gradcam_url = self.url_resolver(gc_file)
        except Exception as e:
            logger.error("Error al generar Grad-CAM: %s", e)
        return gradcam_path, gradcam_url

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        image = self._load_image(source_path)
        if model_name:
            self.classifier.set_active_model(model_name)
        
        breed, score, top_5 = self.classify_detected_dog_with_probs(image)
        ood_detected, ood_score = self.detect_ood(image)
        gradcam_path, gradcam_url = self._generate_gradcam(image, breed, output_path)

        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
            gradcam_path=gradcam_path,
            gradcam_url=gradcam_url,
            ood_detected=ood_detected,
            ood_score=round(float(ood_score), 4),
            top_5=top_5,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        ood_detected, ood_score = self.detect_ood(image)

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score, top_5 = self.classify_detected_dog_with_probs(crop)
            gc_path, gc_url = self._generate_gradcam(crop, breed, output_path)
            
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                    gradcam_path=gc_path,
                    gradcam_url=gc_url,
                    top_5=top_5,
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
            ood_detected=ood_detected,
            ood_score=round(float(ood_score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
