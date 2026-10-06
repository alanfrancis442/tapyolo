"""Mask extents and YOLO normalization from PRD §3.2 and §4.1."""

from __future__ import annotations

import numpy as np


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive ``(x_min, y_min, x_max, y_max)`` of the foreground pixels."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def xyxy_to_yolo(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """Normalize inclusive extents to ``(x_center, y_center, width, height)`` in ``[0, 1]``."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    box_x = (x_min + x_max) / (2.0 * width)
    box_y = (y_min + y_max) / (2.0 * height)
    box_w = (x_max - x_min) / width
    box_h = (y_max - y_min) / height
    return tuple(float(min(1.0, max(0.0, value))) for value in (box_x, box_y, box_w, box_h))  # type: ignore[return-value]


def yolo_to_xyxy(
    box_x: float,
    box_y: float,
    box_w: float,
    box_h: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """Inverse of :func:`xyxy_to_yolo`."""
    x_min = (box_x - box_w / 2.0) * width
    y_min = (box_y - box_h / 2.0) * height
    x_max = (box_x + box_w / 2.0) * width
    y_max = (box_y + box_h / 2.0) * height
    return x_min, y_min, x_max, y_max


def box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """IoU for corner boxes that share one coordinate convention."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union
