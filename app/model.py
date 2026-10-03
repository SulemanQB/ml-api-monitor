import logging
import os
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from PIL import Image

try:
    import torch
except (ImportError, OSError):
    torch = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrediction:
    label: str
    confidence: float
    latency_ms: float


class TinyColorClassifier:
    """Brightness and color heuristic for offline demos. Not a trained model."""

    labels = ("bright-scene", "dark-scene", "colorful-scene")

    def forward(self, batch: np.ndarray) -> np.ndarray:
        means = batch.mean(axis=(1, 2))
        brightness = means.mean(axis=1)
        colorfulness = means.max(axis=1) - means.min(axis=1)
        return np.stack((brightness * 8, (1 - brightness) * 8, colorfulness * 10), axis=1)

    def eval(self) -> "TinyColorClassifier":
        return self


class ImageClassifier:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("MODEL_NAME", "tiny-color")
        self.model: object | None = None
        self.labels: tuple[str, ...] = TinyColorClassifier.labels
        self.transform = None
        self.loaded = False
        self.load_error: str | None = None
        self.load()

    def load(self) -> None:
        try:
            if self.model_name == "mobilenet_v3_small":
                if torch is None:
                    raise RuntimeError("PyTorch native runtime is unavailable")
                from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

                self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
                self.model.eval()
                self.transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
                self.labels = tuple(MobileNet_V3_Small_Weights.DEFAULT.meta["categories"])
            else:
                self.model = TinyColorClassifier().eval()
            self.loaded = True
        except Exception as exc:
            logger.exception("model_load_failed")
            self.load_error = str(exc)
            self.model = None
            self.loaded = False

    @staticmethod
    def _array(image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize((224, 224))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        return array[np.newaxis, ...]

    def predict(self, image: Image.Image) -> ModelPrediction:
        if self.model is None or not self.loaded:
            raise RuntimeError("model is not loaded")
        started = perf_counter()
        if torch is not None and isinstance(self.model, torch.nn.Module):
            tensor = self.transform(image).unsqueeze(0) if self.transform else torch.from_numpy(self._array(image)).permute(0, 3, 1, 2)
            with torch.inference_mode():
                logits = self.model(tensor)[0].numpy()
        else:
            logits = self.model.forward(self._array(image))[0]
        exponentials = np.exp(logits - np.max(logits))
        probabilities = exponentials / exponentials.sum()
        elapsed_ms = (perf_counter() - started) * 1000
        index = int(np.argmax(probabilities))
        return ModelPrediction(self.labels[index], float(probabilities[index]), elapsed_ms)
