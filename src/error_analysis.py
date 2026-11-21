import argparse
import json
import math
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .evaluator import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATASET_PATH,
    EXCLUDED_CLASSES,
    RANDOM_STATE,
    LabeledImage,
    _materialize_zip_test_images,
    build_reproducibility_metadata,
    load_labeled_images,
    reconstruct_notebook_split,
)
from .predictor import CLASS_MAPPING, CLASS_NAMES, EyeDiseasePredictor


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "evaluation"
CONFIDENCE_BINS = np.linspace(0.0, 1.0, 11)
SPLIT_LIMITATION = (
    "The notebook did not save its ordered image list or test indices. For a ZIP "
    "source, the split uses archive member order, which cannot be proven identical "
    "to the original Kaggle filesystem glob order."
)


def _validate_inputs(
    test_samples: Sequence[LabeledImage], scores: Sequence[Sequence[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    if not test_samples:
        raise ValueError("Cannot analyze an empty test set.")

    raw_labels = [sample.label for sample in test_samples]
    if any(
        not isinstance(label, (int, np.integer)) or isinstance(label, bool)
        for label in raw_labels
    ):
        raise ValueError("Test labels must use integer class IDs.")
    labels = np.asarray(raw_labels, dtype=np.int64)
    try:
        score_array = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Scores must have one value per configured class for every test sample."
        ) from exc
    if labels.shape != (len(test_samples),):
        raise ValueError("Each test sample must have one class label.")
    if not np.isin(labels, np.arange(len(CLASS_NAMES))).all():
        raise ValueError("Test labels must use the configured class IDs.")
    if score_array.shape != (len(test_samples), len(CLASS_NAMES)):
        raise ValueError(
            "Scores must have one value per configured class for every test sample."
        )
    if not np.isfinite(score_array).all():
        raise ValueError("Scores must contain only finite values.")
    if (score_array < 0.0).any() or (score_array > 1.0).any():
        raise ValueError("Scores must be between zero and one.")
    if not np.allclose(score_array.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("Each score row must sum to one.")
    return labels, score_array


def _confidence_statistics(values: Sequence[float]) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std_dev": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std_dev": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def build_error_report(
    test_samples: Sequence[LabeledImage],
    scores: Sequence[Sequence[float]],
    reproducibility: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build a path-safe report from labels and the predictor's softmax scores."""
    true_labels, score_array = _validate_inputs(test_samples, scores)
    predicted_labels = np.argmax(score_array, axis=1)
    predicted_confidence = score_array[np.arange(len(score_array)), predicted_labels]
    true_confidence = score_array[np.arange(len(score_array)), true_labels]
    correct_mask = predicted_labels == true_labels

    errors = []
    counts_by_true_class = {name: 0 for name in CLASS_NAMES}
    pair_counts: Dict[Tuple[int, int], int] = {}
    for index, (sample, true_label, predicted_label) in enumerate(
        zip(test_samples, true_labels, predicted_labels)
    ):
        if true_label == predicted_label:
            continue
        true_class = CLASS_MAPPING[int(true_label)]
        predicted_class = CLASS_MAPPING[int(predicted_label)]
        counts_by_true_class[true_class] += 1
        pair = (int(true_label), int(predicted_label))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        filename = PurePosixPath(str(sample.source).replace("\\", "/")).name
        errors.append(
            {
                "dataset_relative_path": f"{true_class}/{filename}",
                "filename": filename,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "prediction_confidence": float(predicted_confidence[index]),
                "true_class_confidence": float(true_confidence[index]),
            }
        )

    correct_count = int(np.count_nonzero(correct_mask))
    error_count = len(errors)
    confusion_pairs = [
        {
            "true_class": CLASS_MAPPING[true_label],
            "predicted_class": CLASS_MAPPING[predicted_label],
            "count": count,
        }
        for (true_label, predicted_label), count in sorted(
            pair_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    confidence_histogram = {
        "bin_edges": CONFIDENCE_BINS.tolist(),
        "correct_fraction_by_bin": _histogram_fractions(
            predicted_confidence[correct_mask]
        ),
        "incorrect_fraction_by_bin": _histogram_fractions(
            predicted_confidence[~correct_mask]
        ),
    }
    report: Dict[str, object] = {
        "classes": list(CLASS_NAMES),
        "class_mapping": {str(index): name for index, name in CLASS_MAPPING.items()},
        "summary": {
            "total_samples": len(test_samples),
            "correct_predictions": correct_count,
            "incorrect_predictions": error_count,
            "overall_error_rate": float(error_count / len(test_samples)),
        },
        "misclassification_counts_by_true_class": counts_by_true_class,
        "most_common_confusion_pairs": confusion_pairs,
        "misclassifications": errors,
        "confidence_statistics": {
            "score_type": "Predicted-class softmax output; not calibrated probability.",
            "correct": _confidence_statistics(predicted_confidence[correct_mask]),
            "incorrect": _confidence_statistics(predicted_confidence[~correct_mask]),
            "histogram": confidence_histogram,
        },
        "confidence_interpretation": (
            "Confidence values are model softmax outputs, not calibrated "
            "probabilities. A high score is not proof that a prediction is correct, "
            "and these distributions do not establish clinical reliability."
        ),
        "split": {
            "random_state": RANDOM_STATE,
            "stratified": True,
            "first_split_test_size": 0.2,
            "second_split_validation_size": 0.25,
            "included_extension": ".jpg",
            "excluded_classes": sorted(EXCLUDED_CLASSES),
            "exact_original_test_membership_confirmed": False,
            "membership_limitation": SPLIT_LIMITATION,
        },
        "reproducibility": reproducibility or {},
    }
    return report


def _histogram_fractions(values: Sequence[float]) -> List[float]:
    if not len(values):
        return [0.0] * (len(CONFIDENCE_BINS) - 1)
    counts, _ = np.histogram(values, bins=CONFIDENCE_BINS)
    return (counts / len(values)).astype(float).tolist()


def _select_gallery_indices(
    test_samples: Sequence[LabeledImage],
    scores: Sequence[Sequence[float]],
) -> List[int]:
    true_labels, score_array = _validate_inputs(test_samples, scores)
    predicted_labels = np.argmax(score_array, axis=1)
    groups: Dict[Tuple[int, int], List[int]] = {}
    for index, (true_label, predicted_label) in enumerate(
        zip(true_labels, predicted_labels)
    ):
        if true_label != predicted_label:
            groups.setdefault((int(true_label), int(predicted_label)), []).append(index)

    selected = []
    prediction_confidence = score_array.max(axis=1)
    for pair in sorted(groups):
        indices = groups[pair]
        median_confidence = float(np.median(prediction_confidence[indices]))
        selected.append(
            min(
                indices,
                key=lambda index: (
                    abs(prediction_confidence[index] - median_confidence),
                    index,
                ),
            )
        )
    return selected


def save_confidence_plot(report: Dict[str, object], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histogram = report["confidence_statistics"]["histogram"]
    edges = histogram["bin_edges"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.stairs(
        histogram["correct_fraction_by_bin"],
        edges,
        label="Correct",
        linewidth=2,
    )
    axis.stairs(
        histogram["incorrect_fraction_by_bin"],
        edges,
        label="Incorrect",
        linewidth=2,
    )
    axis.set(
        xlabel="Predicted-class softmax score",
        ylabel="Fraction within group",
        title="Prediction confidence: correct vs. incorrect",
        xlim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_misclassification_gallery(
    test_samples: Sequence[LabeledImage],
    scores: Sequence[Sequence[float]],
    path_by_source: Dict[str, Path],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    selected_indices = _select_gallery_indices(test_samples, scores)
    columns = 2
    rows = max(1, math.ceil(len(selected_indices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, rows * 5), squeeze=False)
    flattened_axes = list(axes.flat)
    for axis_index, sample_index in enumerate(selected_indices):
        sample = test_samples[sample_index]
        score_row = scores[sample_index]
        predicted_index = int(np.argmax(score_row))
        true_index = int(sample.label)
        path = path_by_source[sample.source]
        with Image.open(path) as image:
            axes.flat[axis_index].imshow(image.convert("RGB"))
        axes.flat[axis_index].set_title(
            f"True: {CLASS_MAPPING[true_index]} | "
            f"Predicted: {CLASS_MAPPING[predicted_index]}\n"
            f"Prediction softmax score: {score_row[predicted_index]:.3f}\n"
            f"File: {PurePosixPath(sample.source.replace(chr(92), '/')).name}",
            fontsize=11,
        )
        axes.flat[axis_index].axis("off")

    for axis in flattened_axes[len(selected_indices):]:
        axis.axis("off")
    if not selected_indices:
        axes[0, 0].text(
            0.5,
            0.5,
            "No misclassified images",
            ha="center",
            va="center",
            fontsize=14,
        )
        axes[0, 0].axis("off")
    figure.suptitle("Representative misclassifications (median score per pair)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_error_analysis_outputs(
    report: Dict[str, object],
    test_samples: Sequence[LabeledImage],
    scores: Sequence[Sequence[float]],
    path_by_source: Dict[str, Path],
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "error_analysis.json"
    confidence_plot_path = output_dir / "confidence_distribution.png"
    gallery_path = output_dir / "misclassification_gallery.png"
    with report_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    save_confidence_plot(report, confidence_plot_path)
    save_misclassification_gallery(
        test_samples, scores, path_by_source, gallery_path
    )
    return report_path, confidence_plot_path, gallery_path


def analyze_dataset(
    dataset_path: Path = DEFAULT_DATASET_PATH,
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
    score_rows = []

    def create_report(path_by_source: Dict[str, Path]) -> Dict[str, object]:
        report = build_error_report(
            test_samples,
            score_rows,
            build_reproducibility_metadata(
                dataset_path, checkpoint_path, samples, predictor.device
            ),
        )
        test_class_counts = {
            name: sum(sample.label == index for sample in test_samples)
            for index, name in enumerate(CLASS_NAMES)
        }
        report["split"]["image_ordering"] = ordering_description
        report["split"]["test_class_counts"] = test_class_counts
        save_error_analysis_outputs(
            report, test_samples, score_rows, path_by_source, output_dir
        )
        return report

    def infer(path_by_source: Dict[str, Path]) -> None:
        for index, sample in enumerate(test_samples, start=1):
            logits = predictor._predict_logits(path_by_source[sample.source])
            if logits.shape != (1, len(CLASS_NAMES)):
                raise ValueError(
                    f"Predictor returned unexpected logits for "
                    f"{sample.source}: {tuple(logits.shape)}"
                )
            scores = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
            score_rows.append([float(score) for score in scores])
            if index % 50 == 0:
                print(f"Analyzed {index}/{len(test_samples)} test images")

    if dataset_path.is_dir():
        test_paths = {sample.source: Path(sample.source) for sample in test_samples}
        infer(test_paths)
        return create_report(test_paths)
    else:
        with tempfile.TemporaryDirectory(prefix="eye-disease-error-analysis-") as temp:
            test_paths = _materialize_zip_test_images(
                dataset_path, test_samples, Path(temp)
            )
            infer(test_paths)
            return create_report(test_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze model errors on the reconstructed notebook test split."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    report = analyze_dataset(
        args.dataset,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )
    summary = report["summary"]
    print(
        f"Samples: {summary['total_samples']}; "
        f"errors: {summary['incorrect_predictions']}; "
        f"error rate: {summary['overall_error_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
