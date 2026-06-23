#!/usr/bin/env python3
"""
Traffic Violation Detection Pipeline
=====================================
Production-oriented, modular pipeline for detecting traffic violations from
still images using OpenCV preprocessing and Ultralytics YOLO (v8/v10).

Supported violation types:
  - Triple riding (3+ persons overlapping a motorcycle bounding box)
  - Stop-line crossing while traffic light is RED

Author: Traffic CV Pipeline
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "ultralytics is required. Install with: pip install ultralytics"
    ) from exc

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

# COCO class IDs used by default YOLO weights (yolov8n.pt / yolov10n.pt).
COCO_CLASS_IDS: Dict[str, int] = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}

# Helmet is not a COCO class; supply a custom-trained model and override this ID.
DEFAULT_HELMET_CLASS_ID: Optional[int] = None

# Low-light threshold on mean grayscale intensity (0-255 scale).
LOW_LIGHT_MEAN_THRESHOLD: float = 60.0

# Minimum overlap ratio (intersection / person_area) to associate a person with a motorcycle.
PERSON_MOTORCYCLE_OVERLAP_THRESHOLD: float = 0.30

# Minimum detection confidence from YOLO to keep a box.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.35

# Hardcoded stop-line ROI as (x, y) polygon vertices in image pixel coordinates.
# Adjust these coordinates to match your camera calibration / scene geometry.
STOP_LINE_ROI_POLYGON: np.ndarray = np.array(
    [
        [320, 420],
        [960, 420],
        [980, 480],
        [300, 480],
    ],
    dtype=np.int32,
)


class ViolationType(str, Enum):
    """Canonical violation labels emitted by the pipeline."""

    TRIPLE_RIDING = "triple_riding"
    STOP_LINE = "stop_line_violation"
    NO_HELMET = "no_helmet"  # reserved for future helmet logic


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box with class metadata."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Bottom-center contact point — useful for stop-line checks."""
        cx = int((self.x1 + self.x2) / 2.0)
        cy = int(self.y2)
        return cx, cy

    def as_xyxy_int(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass
class ViolationRecord:
    """Internal representation of a detected violation before packaging."""

    violation_type: ViolationType
    confidence: float
    subject_box: BoundingBox
    metadata: Dict[str, Any] = field(default_factory=dict)
    license_plate: Optional[str] = None
    plate_confidence: Optional[float] = None


@dataclass
class PipelineConfig:
    """Runtime configuration for the traffic violation pipeline."""

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    low_light_threshold: float = LOW_LIGHT_MEAN_THRESHOLD
    overlap_threshold: float = PERSON_MOTORCYCLE_OVERLAP_THRESHOLD
    stop_line_polygon: np.ndarray = field(
        default_factory=lambda: STOP_LINE_ROI_POLYGON.copy()
    )
    helmet_class_id: Optional[int] = DEFAULT_HELMET_CLASS_ID
    vehicle_class_names: Tuple[str, ...] = ("car", "motorcycle", "bus", "truck")
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    gamma: float = 1.2  # >1.0 brightens mid-tones after CLAHE


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def compute_intersection_area(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Return pixel area of intersection between two axis-aligned boxes."""
    x_left = max(box_a.x1, box_b.x1)
    y_top = max(box_a.y1, box_b.y1)
    x_right = min(box_a.x2, box_b.x2)
    y_bottom = min(box_a.y2, box_b.y2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return float((x_right - x_left) * (y_bottom - y_top))


def compute_overlap_ratio(reference: BoundingBox, candidate: BoundingBox) -> float:
    """
    Simple overlap metric: intersection area divided by candidate area.

    Used to determine whether a person bbox is associated with a motorcycle.
    Returns 0.0 when candidate area is zero or boxes do not overlap.
    """
    candidate_area = candidate.area
    if candidate_area <= 0.0:
        return 0.0
    intersection = compute_intersection_area(reference, candidate)
    return intersection / candidate_area


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Standard Intersection over Union between two bounding boxes."""
    intersection = compute_intersection_area(box_a, box_b)
    union = box_a.area + box_b.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def point_in_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
    """
    Check whether a point lies inside a polygon using cv2.pointPolygonTest.

    polygon: array of shape (N, 2) with integer pixel coordinates.
    """
    if polygon is None or len(polygon) < 3:
        return False
    result = cv2.pointPolygonTest(
        polygon.astype(np.float32),
        (float(point[0]), float(point[1])),
        measureDist=False,
    )
    return result >= 0


# ---------------------------------------------------------------------------
# 1. Image Preprocessing
# ---------------------------------------------------------------------------


class ImagePreprocessor:
    """
    Normalizes illumination for low-light / nighttime traffic camera frames.

    Pipeline:
      1. Estimate scene brightness via mean grayscale intensity.
      2. If below threshold, apply CLAHE on the L channel (LAB space).
      3. Apply gamma correction to lift mid-tones without blowing highlights.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def is_low_light(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Determine whether the frame is low-light based on mean pixel intensity.

        Returns:
            (is_low_light, mean_intensity)
        """
        if image is None or image.size == 0:
            return False, 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mean_intensity = float(np.mean(gray))
        is_low = mean_intensity < self._config.low_light_threshold
        logger.debug(
            "Mean intensity=%.2f, low_light=%s (threshold=%.2f)",
            mean_intensity,
            is_low,
            self._config.low_light_threshold,
        )
        return is_low, mean_intensity

    @staticmethod
    def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization on the L channel.

        Operating in LAB color space preserves chroma while enhancing luminance.
        """
        if image is None or image.size == 0:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_enhanced = clahe.apply(l_channel)

        merged = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """
        Apply gamma correction via a precomputed 256-entry LUT.

        gamma > 1.0 brightens; gamma < 1.0 darkens.
        """
        if image is None or image.size == 0:
            return image

        gamma = max(gamma, 1e-6)
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, table)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full preprocessing entry point.

        Returns:
            (processed_image, preprocessing_metadata)
        """
        metadata: Dict[str, Any] = {
            "low_light_detected": False,
            "mean_intensity": 0.0,
            "clahe_applied": False,
            "gamma_applied": False,
        }

        if image is None or image.size == 0:
            logger.warning("Empty image passed to preprocessor; returning as-is.")
            return image, metadata

        is_low, mean_intensity = self.is_low_light(image)
        metadata["mean_intensity"] = mean_intensity
        metadata["low_light_detected"] = is_low

        processed = image.copy()

        # Always apply gentle normalization in low light; skip in daylight to
        # avoid over-processing well-exposed frames.
        if is_low:
            processed = self.apply_clahe(
                processed,
                clip_limit=self._config.clahe_clip_limit,
                tile_grid_size=self._config.clahe_tile_grid_size,
            )
            metadata["clahe_applied"] = True

            processed = self.apply_gamma_correction(processed, gamma=self._config.gamma)
            metadata["gamma_applied"] = True
            logger.info(
                "Low-light preprocessing applied (mean=%.2f, gamma=%.2f).",
                mean_intensity,
                self._config.gamma,
            )

        return processed, metadata


# ---------------------------------------------------------------------------
# 2. Detection & Violation Logic
# ---------------------------------------------------------------------------


class ViolationDetector:
    """
    Runs YOLO inference and applies domain-specific violation heuristics.

    Responsibilities:
      - Object detection (vehicles, persons, helmets)
      - Triple-riding detection via overlap counting
      - Stop-line violation when traffic_light == 'RED'
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._model = YOLO(config.model_path)
        # Build reverse lookup: class_id -> name
        self._id_to_name: Dict[int, str] = {}
        if hasattr(self._model, "names") and self._model.names:
            self._id_to_name = {int(k): str(v) for k, v in self._model.names.items()}

        logger.info("Loaded YOLO model from '%s'.", config.model_path)

    def _resolve_class_name(self, class_id: int) -> str:
        return self._id_to_name.get(class_id, f"class_{class_id}")

    def detect_objects(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Run YOLO inference and parse results into BoundingBox instances.

        Returns an empty list when no objects are detected or input is invalid.
        """
        if image is None or image.size == 0:
            logger.warning("Empty image passed to detector; skipping inference.")
            return []

        try:
            results = self._model.predict(
                source=image,
                conf=self._config.confidence_threshold,
                verbose=False,
            )
        except Exception:
            logger.exception("YOLO inference failed.")
            return []

        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            logger.info("No objects detected in frame.")
            return []

        boxes: List[BoundingBox] = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = self._resolve_class_name(class_id)

            boxes.append(
                BoundingBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                )
            )

        logger.info("Detected %d object(s).", len(boxes))
        return boxes

    def filter_by_class(
        self,
        boxes: Sequence[BoundingBox],
        class_names: Sequence[str],
    ) -> List[BoundingBox]:
        """Return boxes whose class_name is in class_names (case-insensitive)."""
        targets = {name.lower() for name in class_names}
        return [b for b in boxes if b.class_name.lower() in targets]

    def detect_triple_riding(
        self,
        boxes: Sequence[BoundingBox],
    ) -> List[ViolationRecord]:
        """
        Flag motorcycles that overlap with 3 or more person bounding boxes.

        Overlap is measured as intersection_area / person_area >= threshold.
        """
        violations: List[ViolationRecord] = []

        motorcycles = self.filter_by_class(boxes, ("motorcycle",))
        persons = self.filter_by_class(boxes, ("person",))

        if not motorcycles:
            logger.debug("No motorcycles detected; skipping triple-riding check.")
            return violations

        if not persons:
            logger.debug("No persons detected; skipping triple-riding check.")
            return violations

        for moto in motorcycles:
            overlapping_persons: List[BoundingBox] = []
            overlap_scores: List[float] = []

            for person in persons:
                ratio = compute_overlap_ratio(moto, person)
                if ratio >= self._config.overlap_threshold:
                    overlapping_persons.append(person)
                    overlap_scores.append(ratio)

            if len(overlapping_persons) >= 3:
                avg_overlap = float(np.mean(overlap_scores)) if overlap_scores else 0.0
                confidence = min(1.0, (moto.confidence + avg_overlap) / 2.0)

                violations.append(
                    ViolationRecord(
                        violation_type=ViolationType.TRIPLE_RIDING,
                        confidence=confidence,
                        subject_box=moto,
                        metadata={
                            "rider_count": len(overlapping_persons),
                            "overlap_scores": overlap_scores,
                            "motorcycle_confidence": moto.confidence,
                        },
                    )
                )
                logger.info(
                    "Triple riding detected: motorcycle with %d overlapping persons.",
                    len(overlapping_persons),
                )

        return violations

    def detect_stop_line_violations(
        self,
        boxes: Sequence[BoundingBox],
        traffic_light: str,
    ) -> List[ViolationRecord]:
        """
        Flag vehicles whose bottom-center lies inside the stop-line ROI while RED.

        Args:
            boxes: All detected bounding boxes from the current frame.
            traffic_light: Simulated signal state ('RED', 'GREEN', 'YELLOW').
        """
        violations: List[ViolationRecord] = []

        if traffic_light.upper() != "RED":
            logger.debug(
                "Traffic light is '%s'; stop-line check skipped.", traffic_light
            )
            return violations

        vehicles = self.filter_by_class(boxes, self._config.vehicle_class_names)
        if not vehicles:
            logger.debug("No vehicles detected; skipping stop-line check.")
            return violations

        polygon = self._config.stop_line_polygon
        for vehicle in vehicles:
            contact_point = vehicle.bottom_center
            if point_in_polygon(contact_point, polygon):
                violations.append(
                    ViolationRecord(
                        violation_type=ViolationType.STOP_LINE,
                        confidence=vehicle.confidence,
                        subject_box=vehicle,
                        metadata={
                            "contact_point": contact_point,
                            "traffic_light": traffic_light.upper(),
                            "vehicle_class": vehicle.class_name,
                        },
                    )
                )
                logger.info(
                    "Stop-line violation: %s at %s (conf=%.2f).",
                    vehicle.class_name,
                    contact_point,
                    vehicle.confidence,
                )

        return violations

    def run_violation_checks(
        self,
        boxes: Sequence[BoundingBox],
        traffic_light: str = "GREEN",
    ) -> List[ViolationRecord]:
        """Execute all violation heuristics and merge results."""
        violations: List[ViolationRecord] = []
        violations.extend(self.detect_triple_riding(boxes))
        violations.extend(self.detect_stop_line_violations(boxes, traffic_light))
        return violations


# ---------------------------------------------------------------------------
# 3. Mock OCR / Plate Extraction Hook
# ---------------------------------------------------------------------------


def extract_license_plate(cropped_img: np.ndarray) -> Tuple[str, float]:
    """
    Mock license plate OCR hook.

    In production, replace this with a dedicated ALPR model or cloud OCR service.
    This placeholder inspects image dimensions and returns a deterministic mock
    plate string with a simulated confidence score.

    Args:
        cropped_img: BGR crop of the vehicle's lower half (plate region proxy).

    Returns:
        (plate_text, confidence) where confidence is in [0.0, 1.0].
    """
    if cropped_img is None or cropped_img.size == 0:
        return "UNKNOWN", 0.0

    h, w = cropped_img.shape[:2]
    # Simulate higher confidence for reasonably sized crops.
    area_factor = min(1.0, (h * w) / (10_000.0))
    confidence = round(0.55 + 0.40 * area_factor, 3)

    # Deterministic mock plate derived from crop checksum for reproducibility.
    checksum = int(np.sum(cropped_img.astype(np.uint32))) % 10_000
    mock_plate = f"MH12QZ{checksum:04d}"

    logger.debug(
        "Mock OCR: plate=%s, confidence=%.3f (crop=%dx%d).",
        mock_plate,
        confidence,
        w,
        h,
    )
    return mock_plate, confidence


class LicensePlateService:
    """Crops the vehicle region and delegates to the OCR hook."""

    @staticmethod
    def crop_vehicle_lower_half(
        image: np.ndarray,
        box: BoundingBox,
    ) -> Optional[np.ndarray]:
        """
        Crop the lower 50% of a vehicle bounding box — typical plate location.

        Returns None when the crop would be empty or out of bounds.
        """
        if image is None or image.size == 0:
            return None

        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = box.as_xyxy_int()

        # Clamp to image bounds.
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img))

        if x2 <= x1 or y2 <= y1:
            return None

        mid_y = y1 + (y2 - y1) // 2
        crop = image[mid_y:y2, x1:x2]
        return crop if crop.size > 0 else None

    def enrich_with_plate(
        self,
        image: np.ndarray,
        violation: ViolationRecord,
    ) -> ViolationRecord:
        """Attach mock plate text to a violation record."""
        crop = self.crop_vehicle_lower_half(image, violation.subject_box)
        if crop is None:
            violation.license_plate = "UNKNOWN"
            violation.plate_confidence = 0.0
            return violation

        plate, plate_conf = extract_license_plate(crop)
        violation.license_plate = plate
        violation.plate_confidence = plate_conf
        return violation


# ---------------------------------------------------------------------------
# 4. Output Packaging & Visualization
# ---------------------------------------------------------------------------


class OutputRenderer:
    """Draws detections, ROI, and violation annotations on the output frame."""

    # BGR color constants
    COLOR_VEHICLE = (255, 128, 0)
    COLOR_PERSON = (0, 255, 0)
    COLOR_VIOLATION = (0, 0, 255)
    COLOR_ROI = (0, 255, 255)
    COLOR_HELMET = (255, 0, 255)

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        boxes: Sequence[BoundingBox],
        violations: Sequence[ViolationRecord],
        stop_line_polygon: np.ndarray,
    ) -> np.ndarray:
        """Return a copy of frame with bounding boxes and labels drawn."""
        annotated = frame.copy()

        # Draw stop-line ROI.
        if stop_line_polygon is not None and len(stop_line_polygon) >= 3:
            cv2.polylines(
                annotated,
                [stop_line_polygon.reshape((-1, 1, 2))],
                isClosed=True,
                color=OutputRenderer.COLOR_ROI,
                thickness=2,
            )
            cv2.putText(
                annotated,
                "STOP LINE ROI",
                tuple(stop_line_polygon[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                OutputRenderer.COLOR_ROI,
                1,
                cv2.LINE_AA,
            )

        violation_box_ids = {id(v.subject_box) for v in violations}

        for box in boxes:
            x1, y1, x2, y2 = box.as_xyxy_int()
            name = box.class_name.lower()

            if id(box) in violation_box_ids:
                color = OutputRenderer.COLOR_VIOLATION
            elif name == "person":
                color = OutputRenderer.COLOR_PERSON
            elif name == "helmet":
                color = OutputRenderer.COLOR_HELMET
            else:
                color = OutputRenderer.COLOR_VEHICLE

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{box.class_name} {box.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        for violation in violations:
            x1, y1, x2, y2 = violation.subject_box.as_xyxy_int()
            tag = f"VIOLATION: {violation.violation_type.value}"
            cv2.putText(
                annotated,
                tag,
                (x1, min(y2 + 20, annotated.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                OutputRenderer.COLOR_VIOLATION,
                2,
                cv2.LINE_AA,
            )

        return annotated


class OutputPackager:
    """Builds the structured violation ticket dictionary returned to callers."""

    @staticmethod
    def build_ticket(
        violation: ViolationRecord,
        preprocessing_meta: Dict[str, Any],
        detection_count: int,
    ) -> Dict[str, Any]:
        """Create a single violation ticket payload."""
        return {
            "ticket_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "violation_type": violation.violation_type.value,
            "confidence": {
                "violation": round(violation.confidence, 4),
                "detection": round(violation.subject_box.confidence, 4),
                "license_plate": round(violation.plate_confidence or 0.0, 4),
            },
            "license_plate": violation.license_plate or "UNKNOWN",
            "subject": {
                "class_name": violation.subject_box.class_name,
                "bbox_xyxy": list(violation.subject_box.as_xyxy_int()),
            },
            "metadata": violation.metadata,
            "preprocessing": preprocessing_meta,
            "detection_count": detection_count,
        }


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class TrafficViolationPipeline:
    """
    End-to-end orchestrator tying preprocessing, detection, OCR, and packaging.

    Usage:
        pipeline = TrafficViolationPipeline()
        result = pipeline.process_image(image_bgr, traffic_light="RED")
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.preprocessor = ImagePreprocessor(self.config)
        self.detector = ViolationDetector(self.config)
        self.plate_service = LicensePlateService()
        self.renderer = OutputRenderer()
        self.packager = OutputPackager()

    def process_image(
        self,
        image: np.ndarray,
        traffic_light: str = "GREEN",
    ) -> Dict[str, Any]:
        """
        Run the full pipeline on a single BGR image.

        Args:
            image: Input frame as a NumPy array (H, W, 3) in BGR order.
            traffic_light: Simulated signal state for stop-line logic.

        Returns:
            Structured result dictionary. When violations exist, includes tickets
            and an annotated frame; otherwise returns a clean no-violation payload.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            logger.error("Invalid input image.")
            return {
                "success": False,
                "error": "Invalid or empty input image.",
                "violations_detected": False,
                "tickets": [],
                "annotated_frame": None,
            }

        # Step 1 — Preprocess
        processed, preprocess_meta = self.preprocessor.preprocess(image)

        # Step 2 — Detect objects
        boxes = self.detector.detect_objects(processed)

        # Step 3 — Violation heuristics (gracefully handles empty detections)
        raw_violations = self.detector.run_violation_checks(boxes, traffic_light)

        # Step 4 — Mock plate extraction for each violation
        enriched_violations: List[ViolationRecord] = []
        for violation in raw_violations:
            enriched_violations.append(
                self.plate_service.enrich_with_plate(processed, violation)
            )

        # Step 5 — Package tickets
        tickets = [
            self.packager.build_ticket(v, preprocess_meta, len(boxes))
            for v in enriched_violations
        ]

        # Step 6 — Annotated output frame
        annotated = self.renderer.draw_detections(
            processed,
            boxes,
            enriched_violations,
            self.config.stop_line_polygon,
        )

        result: Dict[str, Any] = {
            "success": True,
            "violations_detected": len(tickets) > 0,
            "ticket_count": len(tickets),
            "tickets": tickets,
            "annotated_frame": annotated,
            "preprocessing": preprocess_meta,
            "detection_count": len(boxes),
        }

        if not tickets:
            logger.info("Pipeline complete: no violations detected.")
        else:
            logger.info("Pipeline complete: %d violation(s) flagged.", len(tickets))

        return result

    def process_image_path(
        self,
        image_path: Union[str, Path],
        traffic_light: str = "GREEN",
    ) -> Dict[str, Any]:
        """Convenience wrapper to load an image from disk and run the pipeline."""
        path = Path(image_path)
        if not path.is_file():
            logger.error("Image file not found: %s", path)
            return {
                "success": False,
                "error": f"File not found: {path}",
                "violations_detected": False,
                "tickets": [],
                "annotated_frame": None,
            }

        image = cv2.imread(str(path))
        if image is None:
            logger.error("Failed to decode image: %s", path)
            return {
                "success": False,
                "error": f"Unable to read image: {path}",
                "violations_detected": False,
                "tickets": [],
                "annotated_frame": None,
            }

        return self.process_image(image, traffic_light=traffic_light)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Minimal CLI for batch/single-image processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Traffic violation detection pipeline (YOLO + OpenCV)."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image (BGR).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics YOLO weights path (yolov8*.pt or yolov10*.pt).",
    )
    parser.add_argument(
        "--traffic-light",
        type=str,
        default="GREEN",
        choices=["RED", "GREEN", "YELLOW"],
        help="Simulated traffic signal state.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_annotated.jpg",
        help="Path to save annotated output image.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="YOLO confidence threshold.",
    )
    args = parser.parse_args()

    config = PipelineConfig(
        model_path=args.model,
        confidence_threshold=args.conf,
    )
    pipeline = TrafficViolationPipeline(config)
    result = pipeline.process_image_path(args.image, traffic_light=args.traffic_light)

    if not result.get("success"):
        logger.error("Pipeline failed: %s", result.get("error"))
        return

    if result.get("annotated_frame") is not None:
        cv2.imwrite(args.output, result["annotated_frame"])
        logger.info("Annotated frame saved to %s", args.output)

    print(f"Violations detected: {result['violations_detected']}")
    for ticket in result.get("tickets", []):
        print(f"  [{ticket['ticket_id']}] {ticket['violation_type']} "
              f"(conf={ticket['confidence']['violation']}, "
              f"plate={ticket['license_plate']})")


if __name__ == "__main__":
    main()
