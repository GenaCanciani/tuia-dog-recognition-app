from __future__ import annotations

import logging
import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradCAM:
    """Implementacion de Grad-CAM para explicabilidad de redes convolucionales en PyTorch."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handlers = []

    def _save_activations(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradients(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0]

    def register_hooks(self) -> None:
        self.remove_hooks()
        h1 = self.target_layer.register_forward_hook(self._save_activations)
        try:
            h2 = self.target_layer.register_full_backward_hook(self._save_gradients)
        except AttributeError:
            h2 = self.target_layer.register_backward_hook(self._save_gradients)
        self.handlers.extend([h1, h2])

    def remove_hooks(self) -> None:
        for h in self.handlers:
            h.remove()
        self.handlers.clear()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray | None:
        self.register_hooks()
        try:
            self.model.zero_grad()
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            score = output[0, class_idx]
            score.backward()

            if self.activations is None or self.gradients is None:
                logger.warning("Grad-CAM: No se capturaron activaciones o gradientes.")
                return None

            gradients = self.gradients.data.cpu().numpy()[0]
            activations = self.activations.data.cpu().numpy()[0]

            # Promedio global de los gradientes (pesos alpha)
            weights = np.mean(gradients, axis=(1, 2))
            
            # Combinacion lineal de activaciones ponderadas
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]

            # Aplicar ReLU (solo nos interesan las caracteristicas con impacto positivo)
            cam = np.maximum(cam, 0)

            # Normalizar entre 0 y 1
            max_val = cam.max()
            if max_val > 0:
                cam = cam / max_val

            return cam
        except Exception as e:
            logger.error("Error al generar Grad-CAM: %s", e)
            return None
        finally:
            self.remove_hooks()


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module | None:
    """Devuelve la ultima capa convolucional recomendada para el modelo dado."""
    name = model_name.lower()
    if "resnet18" in name:
        if hasattr(model, "layer4"):
            return model.layer4[-1]
    elif "cnn_custom" in name or "custom" in name:
        if hasattr(model, "features"):
            try:
                return model.features[7]
            except IndexError:
                pass
            # Fallback buscando hacia atras
            for idx in reversed(range(len(model.features))):
                layer = model.features[idx]
                if hasattr(layer, "conv2") or isinstance(layer, nn.Conv2d):
                    return layer
    return None


def apply_gradcam_overlay(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superpone el mapa de calor de Grad-CAM sobre la imagen original (BGR)."""
    # Cambiar tamaño de cam para que coincida con la imagen original
    cam_resized = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]))
    
    # Generar el mapa de calor en BGR usando JET
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    
    # Superponer con transparencia
    overlay = cv2.addWeighted(heatmap, alpha, image_bgr, 1 - alpha, 0)
    return overlay
