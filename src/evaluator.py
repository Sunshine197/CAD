import argparse
import glob
import hashlib
import json
import platform
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL
import sklearn
import torch
import torchvision
import torch.nn.functional as functional
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .predictor import (
    CLASS_NAMES,
    EVAL_IMAGE_SIZE,
    EVAL_NORMALIZATION_MEAN,
    EVAL_NORMALIZATION_STD,
    EyeDiseasePredictor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset without diabetic.zip"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "best_eye_disease_model.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation"
EXCLUDED_CLASSES = {"diabetic_retinopathy"}
RANDOM_STATE = 42
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class LabeledImage:
    source: str
    label: int


class EvaluationError(RuntimeError):
    """Raised when the dataset cannot be evaluated as the notebook expects."""


def load_labeled_images(dataset_path: Path) -> Tuple[List[LabeledImage], str]:
    """Load JPG paths in notebook class order, retaining the source enumeration order."""
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        class_dirs = [
            path.name for path in dataset_path.iterdir() if path.is_dir()
        ]
        classes = sorted(name for name in class_dirs if name not in EXCLUDED_CLASSES)
        if classes != list(CLASS_NAMES):
            raise EvaluationError(
                f"Expected classes {list(CLASS_NAMES)}, found {classes}."
            )

        samples = []
        for class_name in classes:
            class_paths = glob.glob(str(dataset_path / class_name / "*.jpg"))
            label = CLASS_NAMES.index(class_name)
            samples.extend(LabeledImage(path, label) for path in class_paths)
        return samples, "filesystem glob order within sorted class names"

    if not dataset_path.is_file() or not zipfile.is_zipfile(dataset_path):
        raise EvaluationError(
            f"Dataset must be an extracted directory or ZIP archive: {dataset_path}"
        )

    members_by_class = {name: [] for name in CLASS_NAMES}
    found_classes = set()
    with zipfile.ZipFile(dataset_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename).parts
            if len(parts) != 3 or parts[0] != "dataset":
                continue
            class_name = parts[1]
            if class_name in EXCLUDED_CLASSES:
                continue
            if class_name not in CLASS_NAMES:
                raise EvaluationError(
                    f"Unexpected dataset class in ZIP: {class_name}"
                )
            found_classes.add(class_name)
            if PurePosixPath(info.filename).suffix == ".jpg":
                members_by_class[class_name].append(info.filename)

    if found_classes != set(CLASS_NAMES):
        raise EvaluationError(
            f"Expected classes {list(CLASS_NAMES)}, found {sorted(found_classes)}."
        )

    samples = []
    for class_name in CLASS_NAMES:
        label = CLASS_NAMES.index(class_name)
        samples.extend(
            LabeledImage(member, label) for member in members_by_class[class_name]
        )
    if not samples or any(not members_by_class[name] for name in CLASS_NAMES):
        raise EvaluationError("Each expected class must contain at least one JPG.")
    return samples, "ZIP central-directory order within sorted class names"


def reconstruct_notebook_split(
    samples: Sequence[LabeledImage],
) -> Dict[str, List[LabeledImage]]:
    """Apply the notebook's two stratified train_test_split calls and parameters."""
    paths = [sample.source for sample in samples]
    labels = [sample.label for sample in samples]

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_val_labels,
    )

    def pair(paths_subset, labels_subset):
        return [
            LabeledImage(str(path), int(label))
            for path, label in zip(paths_subset, labels_subset)
        ]

    return {
        "train": pair(train_paths, train_labels),
        "validation": pair(val_paths, val_labels),
        "test": pair(test_paths, test_labels),
    }


def _class_counts(samples: Sequence[LabeledImage]) -> Dict[str, int]:
    return {
        class_name: sum(sample.label == index for sample in samples)
        for index, class_name in enumerate(CLASS_NAMES)
    }


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_member_list_sha256(
    samples: Sequence[LabeledImage], dataset_path: Path
) -> str:
    """Digest ordered dataset-relative member names without saving those names."""
    dataset_path = Path(dataset_path)
    digest = hashlib.sha256()
    for sample in samples:
        if dataset_path.is_dir():
            member_name = Path(sample.source).relative_to(dataset_path).as_posix()
        else:
            parts = PurePosixPath(sample.source).parts
            member_name = PurePosixPath(*parts[1:]).as_posix()
        encoded_name = member_name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, byteorder="big"))
        digest.update(encoded_name)
    return digest.hexdigest()


def build_reproducibility_metadata(
    dataset_path: Path,
    checkpoint_path: Path,
    samples: Sequence[LabeledImage],
    device: torch.device,
) -> Dict[str, object]:
    dataset_path = Path(dataset_path)
    checkpoint_path = Path(checkpoint_path)
    dataset_is_archive = dataset_path.is_file()
    return {
        "dataset_archive_sha256": (
            sha256_file(dataset_path) if dataset_is_archive else None
        ),
        "dataset_size_bytes": (
            dataset_path.stat().st_size if dataset_is_archive else None
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "ordered_dataset_member_list_sha256": ordered_member_list_sha256(
            samples, dataset_path
        ),
        "ordered_dataset_member_count": len(samples),
        "preprocessing": {
            "color_mode": "RGB",
            "resize_size": list(EVAL_IMAGE_SIZE),
            "tensor_conversion": "torchvision.transforms.ToTensor",
            "normalization_mean": list(EVAL_NORMALIZATION_MEAN),
            "normalization_std": list(EVAL_NORMALIZATION_STD),
        },
        "software_versions": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "pillow": PIL.__version__,
        },
        "inference_device": str(device),
    }


def calculate_metrics(
    true_labels: Sequence[int],
    predictions: Sequence[int],
    scores: Sequence[Sequence[float]],
    test_loss: float,
) -> Dict[str, object]:
    """Calculate classification metrics from labels, scores, and logits-based loss."""
    y_true = np.asarray(true_labels, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    y_scores = np.asarray(scores, dtype=np.float64)
    class_ids = np.arange(len(CLASS_NAMES))

    if y_true.ndim != 1 or y_pred.shape != y_true.shape:
        raise ValueError("True labels and predictions must be equally sized 1D arrays.")
    if y_scores.shape != (len(y_true), len(CLASS_NAMES)):
        raise ValueError(
            f"Scores must have shape ({len(y_true)}, {len(CLASS_NAMES)})."
        )
    if not len(y_true):
        raise ValueError("Cannot calculate metrics for an empty test set.")
    if not np.isin(y_true, class_ids).all() or not np.isin(y_pred, class_ids).all():
        raise ValueError("Labels and predictions must use the configured class IDs.")
    if not np.isfinite(y_scores).all():
        raise ValueError("Scores must contain only finite values.")
    if not np.isfinite(test_loss) or test_loss < 0:
        raise ValueError("Test loss must be a finite, non-negative value.")

    matrix = confusion_matrix(y_true, y_pred, labels=class_ids)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_ids,
        zero_division=0,
    )

    per_class = {}
    roc_auc = {}
    for index, class_name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        sensitivity_denominator = tp + fn
        specificity_denominator = tn + fp
        per_class[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "sensitivity": (
                float(tp / sensitivity_denominator)
                if sensitivity_denominator
                else None
            ),
            "specificity": (
                float(tn / specificity_denominator)
                if specificity_denominator
                else None
            ),
        }

        binary_labels = (y_true == index).astype(np.int64)
        if np.unique(binary_labels).size == 2:
            roc_auc[class_name] = float(
                roc_auc_score(binary_labels, y_scores[:, index])
            )
        else:
            roc_auc[class_name] = None

    return {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "test_loss": float(test_loss),
        "per_class": per_class,
        "macro_f1": float(f1_score(y_true, y_pred, labels=class_ids, average="macro")),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=class_ids, average="weighted")
        ),
        "roc_auc_ovr": roc_auc,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _evaluate_paths(
    predictor: EyeDiseasePredictor,
    test_samples: Sequence[LabeledImage],
    path_by_source: Dict[str, Path],
    progress_every: int = 50,
) -> Dict[str, object]:
    true_labels = []
    predictions = []
    all_scores = []
    total_loss = 0.0

    for index, sample in enumerate(test_samples, start=1):
        logits = predictor._predict_logits(path_by_source[sample.source])
        if logits.shape != (1, len(CLASS_NAMES)):
            raise EvaluationError(
                f"Predictor returned unexpected logits for {sample.source}: "
                f"{tuple(logits.shape)}"
            )
        target = torch.tensor([sample.label], dtype=torch.long, device=logits.device)
        total_loss += float(
            functional.cross_entropy(logits, target, reduction="sum").item()
        )
        scores = torch.softmax(logits, dim=1)[0].detach().cpu()
        true_labels.append(sample.label)
        predictions.append(int(torch.argmax(scores).item()))
        all_scores.append([float(score.item()) for score in scores])
        if progress_every and index % progress_every == 0:
            print(f"Evaluated {index}/{len(test_samples)} test images")

    if not test_samples:
        raise EvaluationError("Cannot evaluate an empty test set.")
    return calculate_metrics(
        true_labels,
        predictions,
        all_scores,
        test_loss=total_loss / len(test_samples),
    )


def _materialize_zip_test_images(
    dataset_path: Path,
    test_samples: Sequence[LabeledImage],
    temp_dir: Path,
) -> Dict[str, Path]:
    paths = {}
    with zipfile.ZipFile(dataset_path) as archive:
        for index, sample in enumerate(test_samples):
            suffix = PurePosixPath(sample.source).suffix
            image_path = temp_dir / f"{index:04d}_{CLASS_NAMES[sample.label]}{suffix}"
            with archive.open(sample.source) as source, image_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            paths[sample.source] = image_path
    return paths


def _save_confusion_matrix(matrix: Sequence[Sequence[int]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Test-set confusion matrix",
    )
    threshold = max(max(row) for row in matrix) / 2
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(
                column,
                row,
                str(matrix[row][column]),
                ha="center",
                va="center",
                color="white" if matrix[row][column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_evaluation_outputs(
    results: Dict[str, object], output_dir: Path
) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    confusion_path = output_dir / "confusion_matrix.png"
    with metrics_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    _save_confusion_matrix(results["metrics"]["confusion_matrix"], confusion_path)
    return metrics_path, confusion_path


def evaluate_dataset(
    dataset_path: Path,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: Optional[str] = None,
) -> Dict[str, object]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    samples, ordering_description = load_labeled_images(dataset_path)
    splits = reconstruct_notebook_split(samples)
    test_samples = splits["test"]
    predictor = EyeDiseasePredictor(checkpoint_path, device=device)

    if dataset_path.is_dir():
        test_paths = {
            sample.source: Path(sample.source) for sample in test_samples
        }
        metrics = _evaluate_paths(predictor, test_samples, test_paths)
    else:
        with tempfile.TemporaryDirectory(prefix="eye-disease-evaluation-") as temp:
            test_paths = _materialize_zip_test_images(
                dataset_path, test_samples, Path(temp)
            )
            metrics = _evaluate_paths(predictor, test_samples, test_paths)

    results = {
        "classes": list(CLASS_NAMES),
        "class_mapping": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "checkpoint": checkpoint_path.name,
        "dataset": dataset_path.name,
        "reproducibility": build_reproducibility_metadata(
            dataset_path, checkpoint_path, samples, predictor.device
        ),
        "split": {
            "random_state": RANDOM_STATE,
            "stratified": True,
            "first_split_test_size": 0.2,
            "second_split_validation_size": 0.25,
            "included_extension": ".jpg",
            "excluded_classes": sorted(EXCLUDED_CLASSES),
            "image_ordering": ordering_description,
            "class_counts": {
                name: _class_counts(group) for name, group in splits.items()
            },
            "test_class_counts": _class_counts(test_samples),
            "exact_original_test_membership_confirmed": False,
            "membership_limitation": (
                "The notebook did not save its ordered image list or test indices. "
                "For a ZIP source, the split uses archive member order, which cannot "
                "be proven identical to the original Kaggle filesystem glob order."
            ),
        },
        "metrics": metrics,
        "test_loss_method": (
            "Mean per-sample torch.nn.functional.cross_entropy computed directly "
            "from the predictor's original model logits."
        ),
        "score_note": (
            "Scores are model softmax outputs and are not clinically calibrated "
            "probabilities. This evaluation is not clinical validation."
        ),
    }
    metrics_path, confusion_path = save_evaluation_outputs(results, output_dir)
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved confusion matrix: {confusion_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained eye-disease classifier on a stratified test split."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    results = evaluate_dataset(
        args.dataset,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )
    metrics = results["metrics"]
    print(f"Test images: {metrics['sample_count']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
