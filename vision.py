"""Local YOLO detection and baseline/current image comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from database import STATUS_NOT_DETECTED, STATUS_POSSIBLE, STATUS_UNCHANGED

# Illustrative office categories. YOLOv8n is trained on COCO, which does not
# uniquely identify a specific printer, monitor, or other physical device.
CATEGORY_TO_COCO: dict[str, list[str]] = {
    "printer": [],  # not present in COCO class list
    "monitor": ["tv"],
    "laptop": ["laptop"],
    "keyboard": ["keyboard"],
}

ILLUSTRATIVE_CATEGORIES = tuple(CATEGORY_TO_COCO.keys())

COCO_NOTE = {
    "printer": "YOLOv8 (COCO) has no 'printer' class. Status is driven mainly by visual comparison of the same camera view.",
    "monitor": "Monitors are mapped to the COCO class 'tv' (screens). This is a category match, not a unique device ID.",
    "laptop": "Laptops map to the COCO class 'laptop'. The model cannot confirm it is the same physical laptop.",
    "keyboard": "Keyboards map to the COCO class 'keyboard'. Nearby similar objects can cause false matches.",
}


@dataclass
class ModelState:
    available: bool
    message: str
    model: Any = None


@dataclass
class AnalysisResult:
    status: str
    explanation: str
    detections: list[dict[str, Any]] = field(default_factory=list)
    similarity: float = 0.0
    change_ratio: float = 0.0
    histogram_corr: float = 0.0
    annotated_bgr: np.ndarray | None = None
    error: str | None = None


def read_image_bgr(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        try:
            with Image.open(path) as pil_image:
                rgb = np.array(pil_image.convert("RGB"))
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Could not read image: {path.name}") from exc
    if image is None or image.size == 0:
        raise ValueError(f"Invalid or empty image: {path.name}")
    return image


def decode_uploaded_bytes(data: bytes, filename: str) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"'{filename}' is not a readable image. Use PNG, JPG, JPEG, WEBP, or BMP."
        )
    return image


def save_bgr_image(image: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not success:
        raise ValueError("Failed to encode image for saving.")
    encoded.tofile(str(path))
    return path


def load_yolo_model() -> ModelState:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - import-time environment issue
        return ModelState(
            available=False,
            message=(
                "Ultralytics could not be imported. Live detection is unavailable. "
                f"Details: {exc}"
            ),
        )
    try:
        model = YOLO("yolov8n.pt")
        # Warm-up with a tiny array so first-user latency is clearer, but skip
        # if this environment cannot allocate GPU/CPU backend.
        return ModelState(
            available=True,
            model=model,
            message="YOLOv8n loaded locally (CPU-friendly nano model). Category detection only — not unique asset identity.",
        )
    except Exception as exc:
        return ModelState(
            available=False,
            message=(
                "YOLOv8n could not be downloaded or loaded. Check internet access "
                "for the first model download, disk space, and that PyTorch installed "
                f"correctly. Details: {exc}"
            ),
        )


def compare_views(baseline_bgr: np.ndarray, current_bgr: np.ndarray) -> tuple[float, float, float]:
    width, height = 640, 480
    a = cv2.resize(baseline_bgr, (width, height), interpolation=cv2.INTER_AREA)
    b = cv2.resize(current_bgr, (width, height), interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_a, gray_b)
    change_ratio = float((diff > 28).mean())

    hist_a = cv2.calcHist([gray_a], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [64], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    corr = max(0.0, min(1.0, corr))

    similarity = 0.55 * (1.0 - min(1.0, change_ratio * 1.8)) + 0.45 * corr
    similarity = max(0.0, min(1.0, similarity))
    return similarity, change_ratio, corr


def run_detection(model: Any, image_bgr: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, verbose=False, conf=0.35, imgsz=640, device="cpu")
    annotated = image_bgr.copy()
    detections: list[dict[str, Any]] = []
    if not results:
        return detections, annotated

    result = results[0]
    names = result.names or {}
    boxes = result.boxes
    if boxes is None:
        return detections, annotated

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = str(names.get(cls_id, cls_id))
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        detections.append(
            {
                "name": label,
                "confidence": round(conf, 3),
                "box": [x1, y1, x2, y2],
            }
        )
        color = (11, 79, 108)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated,
            caption,
            (x1 + 3, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return detections, annotated


def evaluate_status(
    category: str,
    detections: list[dict[str, Any]],
    similarity: float,
    change_ratio: float,
) -> tuple[str, str]:
    coco_classes = CATEGORY_TO_COCO.get(category, [])
    detected_names = {d["name"] for d in detections}
    matched = sorted(set(coco_classes) & detected_names)
    coco_note = COCO_NOTE.get(category, "")
    sim_pct = f"{similarity:.0%}"
    change_pct = f"{change_ratio:.0%}"

    def with_caveat(text: str) -> str:
        return (
            f"{text} Similarity to baseline view: {sim_pct}; pixel-change ratio: {change_pct}. "
            f"{coco_note} YOLO reports object categories only. A missing detection is not confirmed theft. "
            "A person must review the evidence before any operational action."
        )

    if not coco_classes:
        if similarity >= 0.74:
            return STATUS_UNCHANGED, with_caveat(
                "The current camera view is visually similar to the baseline. "
                "The expected category is not in the YOLO COCO vocabulary, so this is a scene-similarity result, not object identity."
            )
        if similarity >= 0.48:
            return STATUS_POSSIBLE, with_caveat(
                "The current view differs somewhat from the baseline. This may be lighting, camera angle, clutter, or a real change."
            )
        return STATUS_NOT_DETECTED, with_caveat(
            "The current view differs substantially from the baseline and YOLO cannot confirm this category. Treat as a review item only."
        )

    if matched and similarity >= 0.72:
        return STATUS_UNCHANGED, with_caveat(
            f"YOLO found category match(es): {', '.join(matched)}. The camera view is also similar to the baseline."
        )
    if matched:
        return STATUS_POSSIBLE, with_caveat(
            f"YOLO found category match(es): {', '.join(matched)}, but the view has noticeable visual differences versus baseline."
        )
    if similarity >= 0.72:
        return STATUS_POSSIBLE, with_caveat(
            f"YOLO did not find expected class(es) {', '.join(coco_classes)}, but the scene still looks similar. This is often a model miss, occlusion, or angle issue."
        )
    return STATUS_NOT_DETECTED, with_caveat(
        f"YOLO did not find expected class(es) {', '.join(coco_classes)}, and the view differs from baseline. This is a verification cue, not a confirmed loss event."
    )


def analyze_images(
    model_state: ModelState,
    baseline_path: str | Path,
    current_bgr: np.ndarray,
    category: str,
) -> AnalysisResult:
    try:
        baseline_bgr = read_image_bgr(baseline_path)
    except (FileNotFoundError, ValueError) as exc:
        return AnalysisResult(
            status=STATUS_NOT_DETECTED,
            explanation=str(exc),
            error=str(exc),
        )

    similarity, change_ratio, corr = compare_views(baseline_bgr, current_bgr)
    detections: list[dict[str, Any]] = []
    annotated = current_bgr.copy()

    if not model_state.available or model_state.model is None:
        status, explanation = evaluate_status(category, [], similarity, change_ratio)
        explanation = (
            "Live YOLO detection was not available, so this result uses image comparison only. "
            + explanation
        )
        return AnalysisResult(
            status=status,
            explanation=explanation,
            detections=[],
            similarity=similarity,
            change_ratio=change_ratio,
            histogram_corr=corr,
            annotated_bgr=annotated,
            error=model_state.message,
        )

    try:
        detections, annotated = run_detection(model_state.model, current_bgr)
    except Exception as exc:
        status, explanation = evaluate_status(category, [], similarity, change_ratio)
        return AnalysisResult(
            status=status,
            explanation=(
                f"YOLO inference failed ({exc}). Result uses image comparison only. {explanation}"
            ),
            detections=[],
            similarity=similarity,
            change_ratio=change_ratio,
            histogram_corr=corr,
            annotated_bgr=current_bgr.copy(),
            error=str(exc),
        )

    status, explanation = evaluate_status(category, detections, similarity, change_ratio)
    return AnalysisResult(
        status=status,
        explanation=explanation,
        detections=detections,
        similarity=similarity,
        change_ratio=change_ratio,
        histogram_corr=corr,
        annotated_bgr=annotated,
    )


def detections_to_json(detections: list[dict[str, Any]]) -> str:
    return json.dumps(detections)
