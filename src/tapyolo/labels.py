"""YOLO label files: ``class_id x_center y_center width height``."""

from __future__ import annotations

from pathlib import Path

from tapyolo.bbox import xyxy_to_yolo


def format_yolo_line(
    class_id: int,
    yolo: tuple[float, float, float, float],
) -> str:
    x_center, y_center, width, height = yolo
    return f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def write_yolo_label(
    path: Path,
    class_id: int,
    yolo: tuple[float, float, float, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_yolo_line(class_id, yolo), encoding="utf-8")


def write_empty_label(path: Path) -> None:
    """Background sample: an annotation file with no boxes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def write_tracked_labels(
    label_dir: Path,
    stems: list[str],
    boxes: list[tuple[int, int, int, int] | None],
    class_id: int,
    width: int,
    height: int,
) -> list[Path]:
    """Write one label file per keyframe. Missing boxes become empty files."""
    if len(stems) != len(boxes):
        raise ValueError("stems and boxes must be the same length")
    label_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for stem, box in zip(stems, boxes):
        path = label_dir / f"{stem}.txt"
        if box is None:
            write_empty_label(path)
        else:
            yolo = xyxy_to_yolo(box[0], box[1], box[2], box[3], width, height)
            write_yolo_label(path, class_id, yolo)
        paths.append(path)
    return paths


def clean_label(text: str) -> str:
    label = " ".join(text.strip().split())
    if not label:
        raise ValueError("Label is empty")
    if len(label) > 64:
        raise ValueError("Label is longer than 64 characters")
    return label


def remap_label_file(source: Path, dest: Path, class_id: int) -> None:
    """Copy a label file, rewriting every class id. Empty files stay empty."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        write_empty_label(dest)
        return
    raw = source.read_text(encoding="utf-8").strip()
    if not raw:
        write_empty_label(dest)
        return
    lines: list[str] = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        parts[0] = str(int(class_id))
        lines.append(" ".join(parts))
    dest.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
