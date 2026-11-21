from pathlib import Path
from typing import Dict, Union

import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError

from .model import EyeClassificationModel


CLASS_NAMES = ("cataract", "glaucoma", "normal")
CLASS_MAPPING = {index: name for index, name in enumerate(CLASS_NAMES)}


class PredictorError(Exception):
    """Base exception for predictor failures."""


class CheckpointNotFoundError(PredictorError):
    """Raised when the checkpoint file cannot be found."""


class InvalidImageError(PredictorError):
    """Raised when an input file is not a valid image."""


class IncompatibleCheckpointError(PredictorError):
    """Raised when a checkpoint cannot be loaded into the model."""


class EyeDiseasePredictor:
    """Loads the trained eye-disease model and performs standalone inference."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path, None] = None,
        device: Union[str, torch.device, None] = None,
    ):
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).resolve().parents[1] / "best_eye_disease_model.pth"
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if not self.checkpoint_path.is_file():
            raise CheckpointNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        self.eval_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.model = EyeClassificationModel(num_classes=len(CLASS_NAMES))
        try:
            state_dict = torch.load(
                str(self.checkpoint_path),
                map_location=self.device,
                weights_only=True,
            )
        except (OSError, RuntimeError, EOFError) as exc:
            raise PredictorError(
                f"Unable to read checkpoint '{self.checkpoint_path}': {exc}"
            ) from exc

        if not isinstance(state_dict, dict):
            raise IncompatibleCheckpointError(
                f"Checkpoint '{self.checkpoint_path}' does not contain a model state dict."
            )

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise IncompatibleCheckpointError(
                "The checkpoint is incompatible with the expected ResNet50 model "
                f"and three-class classification head: {exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Open, validate, convert, and preprocess one RGB image."""
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise InvalidImageError(f"Image not found: {path}")

        try:
            with Image.open(path) as image:
                rgb_image = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(f"Invalid or unreadable image '{path}': {exc}") from exc

        return self.eval_transforms(rgb_image)

    def predict(self, image_path: Union[str, Path]) -> Dict[str, object]:
        """Return the predicted class and softmax scores for all three classes."""
        image_tensor = self.preprocess(image_path).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            scores = torch.softmax(logits, dim=1)[0].detach().cpu()

        predicted_index = int(torch.argmax(scores).item())
        score_map = {
            CLASS_MAPPING[index]: float(scores[index].item())
            for index in range(len(CLASS_NAMES))
        }

        return {
            "predicted_class": CLASS_MAPPING[predicted_index],
            "scores": score_map,
        }
