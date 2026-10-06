"""Score a tuned detector on anchor images and on the new class.

``meets_90_map`` is true only when the measured anchor mAP50 is at least 0.90.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tapyolo.bbox import box_iou, yolo_to_xyxy

MAP_TARGET = 0.90
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.25


def meets_map_target(map50: float | None, target: float = MAP_TARGET) -> bool:
    return map50 is not None and map50 >= target


def score_sets(
    ground_truth: list[list[tuple[int, float, float, float, float]]],
    predictions: list[list[tuple[int, float, float, float, float, float]]],
    *,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict:
    """AP at one IoU threshold, macro-averaged over classes that appear in the GT.

    Ground truth items are ``(class_id, x1, y1, x2, y2)``.
    Predictions are ``(class_id, confidence, x1, y1, x2, y2)``.
    """
    if len(ground_truth) != len(predictions):
        raise ValueError("ground truth and predictions must cover the same images")
    classes = sorted({int(box[0]) for image in ground_truth for box in image})
    per_class: dict[int, float] = {}
    total_gt = 0
    total_tp = 0
    for class_id in classes:
        ap, recall, gt_count, tp = _average_precision(ground_truth, predictions, class_id, iou_threshold)
        per_class[class_id] = ap
        total_gt += gt_count
        total_tp += tp
    map50 = float(np.mean(list(per_class.values()))) if per_class else None
    recall = (total_tp / total_gt) if total_gt else None
    return {
        "map50": map50,
        "recall": recall,
        "per_class_ap": per_class,
        "images": len(ground_truth),
        "gt_boxes": total_gt,
    }


def evaluate_retention(
    weights: str | Path,
    anchor_dir: Path,
    novel_dir: Path,
    novel_class_id: int,
    *,
    device: str = "cpu",
    confidence: float = CONFIDENCE_THRESHOLD,
    imgsz: int = 640,
) -> dict:
    """Run ``weights`` on both sets and report measured retention."""
    anchor_gt, anchor_images = _load_split(Path(anchor_dir), class_map=None)
    novel_gt, novel_images = _load_split(Path(novel_dir), class_map={0: novel_class_id})
    anchor_pred = _predict(weights, anchor_images, device, confidence, imgsz)
    novel_pred = _predict(weights, novel_images, device, confidence, imgsz)
    anchor_stats = score_sets(anchor_gt, anchor_pred)
    novel_stats = score_sets(novel_gt, novel_pred)
    anchor_map = anchor_stats["map50"]
    return {
        "measured": True,
        "confidence_threshold": confidence,
        "iou_threshold": IOU_THRESHOLD,
        "imgsz": imgsz,
        "anchor_map50": anchor_map,
        "anchor_recall": anchor_stats["recall"],
        "novel_map50": novel_stats["map50"],
        "novel_recall": novel_stats["recall"],
        "anchor_images": anchor_stats["images"],
        "novel_images": novel_stats["images"],
        "anchor_detection_retained": anchor_stats["recall"] is not None and anchor_stats["recall"] > 0,
        "meets_90_map": meets_map_target(anchor_map),
    }


def _average_precision(
    ground_truth,
    predictions,
    class_id: int,
    iou_threshold: float,
) -> tuple[float, float, int, int]:
    npos = sum(1 for image in ground_truth for box in image if int(box[0]) == class_id)
    if npos == 0:
        return 0.0, 0.0, 0, 0
    ranked: list[tuple[float, int, tuple[float, float, float, float]]] = []
    for image_index, image_preds in enumerate(predictions):
        for pred in image_preds:
            if int(pred[0]) != class_id:
                continue
            ranked.append((float(pred[1]), image_index, (pred[2], pred[3], pred[4], pred[5])))
    ranked.sort(key=lambda item: item[0], reverse=True)
    matched = [set() for _ in ground_truth]
    tp: list[int] = []
    fp: list[int] = []
    for _confidence, image_index, box in ranked:
        candidates = [item for item in ground_truth[image_index] if int(item[0]) == class_id]
        best_iou = 0.0
        best_index = None
        for gt_index, gt in enumerate(candidates):
            if gt_index in matched[image_index]:
                continue
            overlap = box_iou(box, (gt[1], gt[2], gt[3], gt[4]))
            if overlap > best_iou:
                best_iou = overlap
                best_index = gt_index
        if best_index is not None and best_iou >= iou_threshold:
            matched[image_index].add(best_index)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if not ranked:
        return 0.0, 0.0, npos, 0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / float(npos)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = _integrate_ap(recall, precision)
    final_tp = int(tp_cum[-1])
    return ap, final_tp / float(npos), npos, final_tp


def _integrate_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def _load_split(root: Path, class_map: dict[int, int] | None):
    import cv2

    image_dir = root / "images"
    images: list[np.ndarray] = []
    ground_truth: list[list[tuple[int, float, float, float, float]]] = []
    if not image_dir.is_dir():
        return ground_truth, images
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read {image_path}")
        height, width = image.shape[:2]
        label_path = root / "labels" / f"{image_path.stem}.txt"
        ground_truth.append(_read_label(label_path, width, height, class_map))
        images.append(image)
    return ground_truth, images


def _read_label(path: Path, width: int, height: int, class_map: dict[int, int] | None):
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    boxes = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_map is not None:
            class_id = class_map.get(class_id, class_id)
        x_min, y_min, x_max, y_max = yolo_to_xyxy(
            float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), width, height
        )
        boxes.append((class_id, x_min, y_min, x_max, y_max))
    return boxes


def _predict(weights, images, device: str, confidence: float, imgsz: int):
    if not images:
        return []
    from ultralytics import YOLO

    model = YOLO(str(weights))
    predictions = []
    for image in images:
        result = model.predict(image, device=device, conf=confidence, imgsz=imgsz, verbose=False)[0]
        boxes = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for class_id, conf, coords in zip(classes, confs, xyxy):
                x1, y1, x2, y2 = (float(v) for v in coords.tolist())
                boxes.append((int(class_id), float(conf), x1, y1, x2, y2))
        predictions.append(boxes)
    return predictions
