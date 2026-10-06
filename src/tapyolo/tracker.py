"""Bidirectional box propagation across keyframes.

ByteTrack associates detections; it does not invent a box from a single tap.
This module proposes a box on each keyframe with template matching and an
IoU/motion gate, running backward to frame 0 and forward to the end. When
``supervision.ByteTrack`` can keep one identity across those proposals, the
result is reported as ``bytetrack``. Otherwise the motion boxes stand alone.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from tapyolo.bbox import box_iou


@dataclass
class PropagationResult:
    boxes: list[tuple[int, int, int, int] | None]
    scores: list[float]
    backend: str


@dataclass
class _Template:
    gray: np.ndarray
    pixel_w: int
    pixel_h: int
    anchor_x: int
    anchor_y: int


def propagate_bidirectional(
    frames: list[np.ndarray],
    start_index: int,
    bbox: tuple[int, int, int, int],
) -> PropagationResult:
    if not frames:
        raise ValueError("no frames to track")
    if not 0 <= start_index < len(frames):
        raise ValueError("start index is outside the keyframe list")
    height, width = frames[0].shape[:2]
    for frame in frames:
        if frame.shape[0] != height or frame.shape[1] != width:
            raise ValueError("frames differ in size")
    box = _validate_box(bbox, width, height)
    template = _build_template(frames[start_index], box)
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(frames)
    scores = [0.0] * len(frames)
    boxes[start_index] = box
    scores[start_index] = 1.0
    _track_direction(frames, start_index, box, template, +1, boxes, scores)
    _track_direction(frames, start_index, box, template, -1, boxes, scores)
    backend = _confirm_bytetrack(boxes, scores)
    return PropagationResult(boxes, scores, backend)


def _track_direction(
    frames: list[np.ndarray],
    start_index: int,
    box: tuple[int, int, int, int],
    template: _Template,
    direction: int,
    boxes: list[tuple[int, int, int, int] | None],
    scores: list[float],
) -> None:
    height, width = frames[0].shape[:2]
    cx, cy = _center(box)
    vx, vy = 0.0, 0.0
    index = start_index
    while True:
        index += direction
        if index < 0 or index >= len(frames):
            break
        pred_cx = cx + vx
        pred_cy = cy + vy
        gray = _gray_u8(frames[index])
        matched = _match(gray, template, (pred_cx, pred_cy), (vx, vy))
        predicted = _shift_box(box if boxes[index - direction] is None else boxes[index - direction], vx, vy, width, height)
        if matched is None:
            chosen, score = predicted, 0.0
        else:
            found, score = matched
            overlap = box_iou(found, predicted)
            if score >= 0.55 or (score >= 0.4 and overlap >= 0.05):
                chosen = found
            else:
                chosen, score = predicted, 0.0
        chosen = _validate_box(chosen, width, height)
        new_cx, new_cy = _center(chosen)
        if score > 0:
            vx = new_cx - cx
            vy = new_cy - cy
        cx, cy = new_cx, new_cy
        boxes[index] = chosen
        scores[index] = score


def _build_template(frame: np.ndarray, box: tuple[int, int, int, int]) -> _Template:
    """Pad the box so a flat object still has an edge for correlation."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad = 4
    tx1 = max(0, x1 - pad)
    ty1 = max(0, y1 - pad)
    tx2 = min(width - 1, x2 + pad)
    ty2 = min(height - 1, y2 + pad)
    gray = _gray_u8(frame)
    patch = gray[ty1 : ty2 + 1, tx1 : tx2 + 1]
    if patch.size == 0 or float(patch.std()) < 1e-3:
        raise ValueError("template has no texture to track")
    return _Template(
        gray=patch,
        pixel_w=x2 - x1 + 1,
        pixel_h=y2 - y1 + 1,
        anchor_x=x1 - tx1,
        anchor_y=y1 - ty1,
    )


def _match(
    gray: np.ndarray,
    template: _Template,
    pred_center: tuple[float, float],
    velocity: tuple[float, float],
) -> tuple[tuple[int, int, int, int], float] | None:
    cv2 = _cv2()
    height, width = gray.shape
    vx, vy = velocity
    pcx, pcy = pred_center
    scales = (0.9, 1.0, 1.1)
    best: tuple[float, float, tuple[int, int, int, int]] | None = None
    base = template.gray
    for scale in scales:
        tw = max(1, int(round(base.shape[1] * scale)))
        th = max(1, int(round(base.shape[0] * scale)))
        if th >= height or tw >= width:
            continue
        patch = base if scale == 1.0 else cv2.resize(base, (tw, th), interpolation=cv2.INTER_LINEAR)
        if float(patch.std()) < 1e-3:
            continue
        rad_x = max(tw, template.pixel_w) + int(abs(vx)) + 24
        rad_y = max(th, template.pixel_h) + int(abs(vy)) + 24
        x0 = int(min(max(0, pcx - rad_x), width - 1))
        y0 = int(min(max(0, pcy - rad_y), height - 1))
        x1 = int(min(width, max(x0 + tw + 1, pcx + rad_x)))
        y1 = int(min(height, max(y0 + th + 1, pcy + rad_y)))
        roi = gray[y0:y1, x0:x1]
        if roi.shape[0] < th or roi.shape[1] < tw:
            roi = gray
            x0, y0 = 0, 0
        result = cv2.matchTemplate(roi, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if not np.isfinite(max_val):
            continue
        mx, my = x0 + max_loc[0], y0 + max_loc[1]
        box_x = int(round(mx + template.anchor_x * scale))
        box_y = int(round(my + template.anchor_y * scale))
        box_w = max(1, int(round(template.pixel_w * scale)))
        box_h = max(1, int(round(template.pixel_h * scale)))
        found = (
            box_x,
            box_y,
            box_x + box_w - 1,
            box_y + box_h - 1,
        )
        if best is None or max_val > best[0] + 0.05 or (
            abs(max_val - best[0]) <= 0.05 and abs(scale - 1.0) < best[1]
        ):
            best = (float(max_val), abs(scale - 1.0), found)
    if best is None:
        return None
    return best[2], best[0]


def _confirm_bytetrack(
    boxes: list[tuple[int, int, int, int] | None],
    scores: list[float],
) -> str:
    try:
        import supervision as sv
    except Exception:
        return "iou-motion"
    kwargs = {
        "minimum_matching_threshold": 0.9,
        "minimum_consecutive_frames": 1,
        "lost_track_buffer": max(30, len(boxes)),
    }
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            forward_tracker = sv.ByteTrack(**kwargs)
            backward_tracker = sv.ByteTrack(**kwargs)
        forward = _run_bytetrack(sv, forward_tracker, boxes, scores)
        backward = _run_bytetrack(sv, backward_tracker, list(reversed(boxes)), list(reversed(scores)))
    except Exception:
        return "iou-motion"
    present = sum(box is not None for box in boxes)
    if present == 0:
        return "iou-motion"
    covered = max(forward, backward)
    if covered / present >= 0.9:
        return "bytetrack"
    return "iou-motion"


def _run_bytetrack(sv, tracker, boxes, scores) -> int:
    update = getattr(tracker, "update_with_detections", None)
    if update is None:
        return 0
    covered = 0
    for box, score in zip(boxes, scores):
        if box is None:
            empty = sv.Detections(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                confidence=np.zeros((0,), dtype=np.float32),
            )
            tracked = update(empty)
        else:
            x1, y1, x2, y2 = box
            detections = sv.Detections(
                xyxy=np.array([[x1, y1, x2 + 1, y2 + 1]], dtype=np.float32),
                confidence=np.array([max(float(score), 0.01)], dtype=np.float32),
            )
            tracked = update(detections)
        ids = getattr(tracked, "tracker_id", None)
        if ids is not None and len(ids) > 0 and ids[0] is not None:
            covered += 1
    return covered


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _shift_box(
    box: tuple[int, int, int, int],
    dx: float,
    dy: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pixel_w = x2 - x1
    pixel_h = y2 - y1
    nx1 = int(round(x1 + dx))
    ny1 = int(round(y1 + dy))
    nx1 = min(max(0, nx1), max(0, width - pixel_w - 1))
    ny1 = min(max(0, ny1), max(0, height - pixel_h - 1))
    return nx1, ny1, nx1 + pixel_w, ny1 + pixel_h


def _validate_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    if width <= 0 or height <= 0:
        raise ValueError("frame size is empty")
    x1 = min(max(0, x1), width - 1)
    y1 = min(max(0, y1), height - 1)
    x2 = min(max(0, x2), width - 1)
    y2 = min(max(0, y2), height - 1)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _gray_u8(image: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _cv2():
    import cv2

    return cv2
