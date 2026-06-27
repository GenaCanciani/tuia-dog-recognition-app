from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime
from torch.utils.data import DataLoader
from torchvision import datasets, models as tv_models, transforms

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        shortcut: list[nn.Module] = []
        if stride != 1 or in_channels != out_channels:
            shortcut = [
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, embed_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ClassifierService:
    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _get_num_classes(self) -> int:
        train_dir = self.dataset_path / "train"
        if not train_dir.is_dir():
            raise ValueError(f"Train directory not found: {train_dir}")
        return len([d for d in train_dir.iterdir() if d.is_dir()])

    def _build_model(self) -> nn.Module:
        num_classes = self._get_num_classes()
        if self.active_model_name == "resnet18_finetuned":
            model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif self.active_model_name == "cnn_custom":
            model = CustomCNN(num_classes, embed_dim=512)
        else:
            raise ValueError(f"Unknown model: {self.active_model_name}")
        return model

    def _add_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            sigma = torch.empty(1).uniform_(0.01, 0.05).item()
            noise = torch.randn_like(tensor) * sigma
            tensor = tensor + noise
        return tensor

    def _get_transforms(self, augment: bool = False) -> transforms.Compose:
        pipeline = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        if augment:
            pipeline.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ])
        pipeline.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if augment:
            pipeline.append(transforms.Lambda(self._add_gaussian_noise))
        return transforms.Compose(pipeline)

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        logger.info("Entrenando modelo: %s", self.active_model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Dispositivo: %s", device)

        model = self._build_model().to(device)
        num_classes = model.classifier.out_features if self.active_model_name == "cnn_custom" else model.fc.out_features

        train_ds = datasets.ImageFolder(
            str(self.dataset_path / "train"), transform=self._get_transforms(augment=True)
        )
        valid_ds = datasets.ImageFolder(
            str(self.dataset_path / "valid"), transform=self._get_transforms(augment=False)
        )

        # Pesos por clase para balancear
        samples_per_class = [len(train_ds.imgs) // num_classes] * num_classes
        counts = [0] * num_classes
        for _, y in train_ds.samples:
            counts[y] += 1
        class_weights = torch.tensor(
            [len(train_ds.samples) / (num_classes * max(c, 1)) for c in counts],
            dtype=torch.float32,
        ).to(device)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=0)

        lr = 1e-4 if self.active_model_name == "resnet18_finetuned" else 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        epochs = 15
        best_loss = float("inf")

        self._train_history = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": [],
        }

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            model.eval()
            valid_loss = 0.0
            valid_correct = 0
            valid_total = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    valid_correct += (preds == labels).sum().item()
                    valid_total += labels.size(0)

            valid_loss /= valid_total
            valid_acc = valid_correct / valid_total
            scheduler.step(valid_loss)

            logger.info(
                "Epoch %2d/%d — train_loss=%.4f train_acc=%.4f — valid_loss=%.4f valid_acc=%.4f",
                epoch, epochs, train_loss, train_acc, valid_loss, valid_acc,
            )

            self._train_history["train_loss"].append(train_loss)
            self._train_history["train_acc"].append(train_acc)
            self._train_history["valid_loss"].append(valid_loss)
            self._train_history["valid_acc"].append(valid_acc)

            if valid_loss < best_loss:
                best_loss = valid_loss
                self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                model.cpu()
                torch.save(model, str(self.active_checkpoint))
                model.to(device)
                logger.info("Checkpoint guardado: %s (valid_loss=%.4f)", self.active_checkpoint, valid_loss)

        model.cpu()
        torch.save(model, str(self.active_checkpoint))
        logger.info("Entrenamiento completado. Checkpoint final: %s", self.active_checkpoint)

    def evaluate_classifier(self) -> dict[str, float]:
        logger.info("Evaluando modelo: %s", self.active_model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.load_model().to(device)
        model.eval()

        test_ds = datasets.ImageFolder(
            str(self.dataset_path / "test"), transform=self._get_transforms(augment=False)
        )
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

        all_preds: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())

        preds_t = torch.tensor(all_preds)
        labels_t = torch.tensor(all_labels)
        num_classes = len(test_ds.classes)

        accuracy = (preds_t == labels_t).float().mean().item()

        # Métricas por clase y macro-promedio
        precisions, recalls, specificities, f1s = [], [], [], []
        for c in range(num_classes):
            tp = ((preds_t == c) & (labels_t == c)).sum().item()
            fp = ((preds_t == c) & (labels_t != c)).sum().item()
            fn = ((preds_t != c) & (labels_t == c)).sum().item()
            tn = ((preds_t != c) & (labels_t != c)).sum().item()

            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

            precisions.append(prec)
            recalls.append(rec)
            specificities.append(spec)
            f1s.append(f1)

        self._eval_preds = all_preds
        self._eval_labels = all_labels
        self._eval_classes = test_ds.classes

        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(float(np.mean(precisions)), 4),
            "recall": round(float(np.mean(recalls)), 4),
            "specificity": round(float(np.mean(specificities)), 4),
            "f1": round(float(np.mean(f1s)), 4),
        }
        logger.info("Métricas: %s", metrics)
        return metrics

