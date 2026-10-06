"""Local rehearsal buffer of 25 multi-class images.

These stand in for a COCO replay split so training can run without downloading
COCO. Pass an existing anchor directory to ``train_detection_head`` to use
real images instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tapyolo.bbox import xyxy_to_yolo

ANCHOR_NAMES = ["person", "bicycle", "car", "cup", "chair"]
ANCHOR_COUNT = 25


def generate_anchor_set(dest: Path, count: int = ANCHOR_COUNT) -> Path:
    """Write ``count`` JPEG images and YOLO labels under ``dest``."""
    cv2 = _cv2()
    images = dest / "images"
    labels = dest / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    colors = (
        (180, 80, 60),
        (40, 160, 200),
        (30, 90, 220),
        (80, 180, 70),
        (200, 160, 40),
    )
    for index in range(count):
        canvas = np.full((height, width, 3), 32, np.uint8)
        class_id = index % len(ANCHOR_NAMES)
        x = 16 + (index % 5) * 8
        y = 20 + (index % 4) * 6
        bw, bh = 72, 64
        _paint(canvas, x, y, bw, bh, colors[class_id])
        lines = [_line(class_id, x, y, bw, bh, width, height)]
        if index % 2 == 0:
            other = (class_id + 1) % len(ANCHOR_NAMES)
            ox, oy, ow, oh = 180, 130, 90, 58
            _paint(canvas, ox, oy, ow, oh, colors[other])
            lines.append(_line(other, ox, oy, ow, oh, width, height))
        stem = f"anchor_{index:03d}"
        image_path = images / f"{stem}.jpg"
        if not cv2.imwrite(str(image_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
            raise RuntimeError(f"failed to write {image_path}")
        label_path = labels / f"{stem}.txt"
        label_path.write_text("".join(lines), encoding="utf-8")
    (dest / "classes.txt").write_text("\n".join(ANCHOR_NAMES) + "\n", encoding="utf-8")
    return dest


def _paint(canvas: np.ndarray, x: int, y: int, bw: int, bh: int, color: tuple[int, int, int]) -> None:
    canvas[y : y + bh, x : x + bw] = color
    canvas[y + 8 : y + bh - 8, x + 8 : x + 16] = (250, 250, 250)


def _line(class_id: int, x: int, y: int, bw: int, bh: int, width: int, height: int) -> str:
    yolo = xyxy_to_yolo(x, y, x + bw - 1, y + bh - 1, width, height)
    return write_line(class_id, yolo)


def write_line(class_id: int, yolo: tuple[float, float, float, float]) -> str:
    x_center, y_center, box_w, box_h = yolo
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"


def _cv2():
    import cv2

    return cv2
