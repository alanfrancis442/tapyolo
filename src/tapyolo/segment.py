"""Point-prompt segmentation: MobileSAM ONNX when weights exist, else flood fill / GrabCut."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tapyolo.bbox import mask_to_bbox, xyxy_to_yolo

SAM_INPUT = 1024
ENCODER_NAMES = ("mobile_sam_encoder.onnx", "mobile_sam.encoder.onnx")
DECODER_NAMES = ("mobile_sam_decoder.onnx", "mobile_sam.decoder.onnx")


class FallbackSegmenter:
    """Deterministic tap segmenter.

    Flood fill from the tap is used when the region is a plausible object.
    GrabCut around the tap runs when the fill covers almost nothing or almost
    the whole frame. A fixed window is the last resort so the pipeline still
    emits a box.
    """

    name = "fallback"

    def segment(self, image: np.ndarray, x: int, y: int) -> tuple[np.ndarray, str]:
        cv2 = _cv2()
        frame = np.ascontiguousarray(image)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        height, width = frame.shape[:2]
        sx, sy = _clip_point(x, y, width, height)
        filled = _flood(frame, sx, sy, tolerance=32)
        if _usable(filled):
            return filled, "flood"
        cut = _grabcut(frame, sx, sy)
        if _usable(cut):
            return cut, "grabcut"
        wide = _flood(frame, sx, sy, tolerance=64)
        if _usable(wide):
            return wide, "flood"
        return _window_mask(height, width, sx, sy, half=16), "box"


class OnnxMobileSamSegmenter:
    """MobileSAM encoder + decoder ONNX pair.

    Expected files follow the common SAM export: an image encoder that emits
    ``image_embeddings``, and a decoder that consumes embeddings, point
    coordinates, point labels, a mask prompt, and the original size.
    """

    name = "onnx-mobilesam"

    def __init__(self, encoder_path: Path, decoder_path: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = _providers(ort)
        self.encoder = ort.InferenceSession(str(encoder_path), sess_options=options, providers=providers)
        self.decoder = ort.InferenceSession(str(decoder_path), sess_options=options, providers=providers)
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path

    def segment(self, image: np.ndarray, x: int, y: int) -> tuple[np.ndarray, str]:
        cv2 = _cv2()
        frame = image
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        height, width = frame.shape[:2]
        sx, sy = _clip_point(x, y, width, height)
        input_size = _spatial_size(self.encoder)
        tensor, scale = prepare_encoder_input(frame, input_size)
        enc_input = self.encoder.get_inputs()[0]
        embedding = self.encoder.run(None, {enc_input.name: tensor})[0]
        point = np.array([[[sx * scale, sy * scale]]], dtype=np.float32)
        labels = np.array([[1]], dtype=np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask = np.zeros((1,), dtype=np.float32)
        orig = np.array([height, width], dtype=np.float32)
        feed = _decoder_feed(
            self.decoder,
            embedding=embedding,
            point_coords=point,
            point_labels=labels,
            mask_input=mask_input,
            has_mask_input=has_mask,
            orig_im_size=orig,
        )
        outputs = self.decoder.run(None, feed)
        mask = postprocess_sam_mask(outputs[0], outputs[1] if len(outputs) > 1 else None, (height, width))
        return mask, "onnx"


@dataclass
class SegmenterChoice:
    segmenter: FallbackSegmenter | OnnxMobileSamSegmenter
    onnx_error: str | None = None

    @property
    def name(self) -> str:
        return self.segmenter.name


@dataclass
class SegmentOutput:
    mask: np.ndarray
    bbox: tuple[int, int, int, int] | None
    yolo: tuple[float, float, float, float] | None
    elapsed_ms: float
    backend: str
    detail: str


def prepare_encoder_input(image: np.ndarray, input_size: int = SAM_INPUT) -> tuple[np.ndarray, float]:
    """RGB, longest-side resize, pad to square, SAM mean/std. Returns NCHW and the scale."""
    cv2 = _cv2()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    height, width = rgb.shape[:2]
    scale = float(input_size) / float(max(height, width))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((input_size, input_size, 3), dtype=np.float32)
    canvas[:resized_h, :resized_w] = resized.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    canvas = (canvas - mean) / std
    tensor = np.transpose(canvas, (2, 0, 1))[None].astype(np.float32)
    return tensor, scale


def scale_point(x: float, y: float, scale: float) -> tuple[float, float]:
    return x * scale, y * scale


def postprocess_sam_mask(
    masks: np.ndarray,
    ious: np.ndarray | None,
    original_hw: tuple[int, int],
) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        index = 0
        if ious is not None:
            scores = np.asarray(ious).reshape(-1)
            if scores.size:
                index = int(np.argmax(scores))
        arr = arr[min(index, arr.shape[0] - 1)]
    if arr.ndim != 2:
        raise ValueError(f"unexpected mask rank {masks.shape}")
    binary = (arr > 0.0).astype(np.uint8)
    height, width = original_hw
    if binary.shape != (height, width):
        cv2 = _cv2()
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    return (binary > 0).astype(np.uint8)


def find_onnx_weights(roots: list[Path] | None = None) -> tuple[Path, Path] | None:
    encoder_env = os.environ.get("TAPYOLO_MOBILESAM_ENCODER")
    decoder_env = os.environ.get("TAPYOLO_MOBILESAM_DECODER")
    if encoder_env and decoder_env:
        encoder, decoder = Path(encoder_env), Path(decoder_env)
        if encoder.is_file() and decoder.is_file():
            return encoder, decoder
    for root in roots or default_weight_roots():
        for enc_name in ENCODER_NAMES:
            for dec_name in DECODER_NAMES:
                encoder, decoder = root / enc_name, root / dec_name
                if encoder.is_file() and decoder.is_file():
                    return encoder, decoder
    return None


def default_weight_roots() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        Path.cwd() / "weights",
        here / "weights",
        Path.home() / ".cache" / "tapyolo" / "weights",
    ]


def load_segmenter(roots: list[Path] | None = None) -> SegmenterChoice:
    found = find_onnx_weights(roots)
    if found is None:
        return SegmenterChoice(FallbackSegmenter(), onnx_error="weights not found")
    try:
        return SegmenterChoice(OnnxMobileSamSegmenter(found[0], found[1]), onnx_error=None)
    except Exception as exc:
        return SegmenterChoice(FallbackSegmenter(), onnx_error=str(exc))


def segment_point(
    image: np.ndarray,
    x: int,
    y: int,
    segmenter: FallbackSegmenter | OnnxMobileSamSegmenter,
) -> SegmentOutput:
    """Time the successful segmenter call. A failed ONNX call is not reported as ONNX latency."""
    started = time.perf_counter()
    try:
        mask, detail = segmenter.segment(image, x, y)
        backend = segmenter.name
    except Exception:
        if segmenter.name == "fallback":
            raise
        started = time.perf_counter()
        mask, detail = FallbackSegmenter().segment(image, x, y)
        backend = "fallback"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    bbox = mask_to_bbox(mask)
    height, width = image.shape[:2]
    yolo = None if bbox is None else xyxy_to_yolo(*bbox, width, height)
    return SegmentOutput(mask, bbox, yolo, elapsed_ms, backend, detail)


def _clip_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return int(min(max(0, x), width - 1)), int(min(max(0, y), height - 1))


def _usable(mask: np.ndarray) -> bool:
    area = float(mask.sum())
    if area < 4:
        return False
    fraction = area / float(mask.size)
    return 0.002 <= fraction <= 0.60


def _flood(image: np.ndarray, x: int, y: int, tolerance: int) -> np.ndarray:
    cv2 = _cv2()
    height, width = image.shape[:2]
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    lo = (tolerance, tolerance, tolerance)
    cv2.floodFill(image.copy(), flood_mask, (x, y), (0, 0, 0), lo, lo, flags)
    return (flood_mask[1:-1, 1:-1] > 0).astype(np.uint8)


def _grabcut(image: np.ndarray, x: int, y: int) -> np.ndarray:
    cv2 = _cv2()
    height, width = image.shape[:2]
    half_w = max(8, int(width * 0.15))
    half_h = max(8, int(height * 0.15))
    x0 = max(0, x - half_w)
    y0 = max(0, y - half_h)
    rect_w = min(width - x0, half_w * 2)
    rect_h = min(height - y0, half_h * 2)
    if rect_w < 2 or rect_h < 2:
        return np.zeros((height, width), np.uint8)
    mask = np.zeros((height, width), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image.copy(), mask, (x0, y0, rect_w, rect_h), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return _component_at(binary, x, y)


def _component_at(binary: np.ndarray, x: int, y: int) -> np.ndarray:
    cv2 = _cv2()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    label = int(labels[y, x])
    if label > 0:
        return (labels == label).astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return (labels == best).astype(np.uint8)


def _window_mask(height: int, width: int, x: int, y: int, half: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    x1, x2 = max(0, x - half), min(width, x + half)
    y1, y2 = max(0, y - half), min(height, y + half)
    mask[y1:y2, x1:x2] = 1
    return mask


def _spatial_size(session) -> int:
    shape = session.get_inputs()[0].shape
    if len(shape) >= 3 and isinstance(shape[-1], int) and shape[-1] > 0:
        return int(shape[-1])
    return SAM_INPUT


def _providers(ort) -> list[str]:
    available = set(ort.get_available_providers())
    chosen: list[str] = []
    if "CUDAExecutionProvider" in available:
        chosen.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        chosen.append("CPUExecutionProvider")
    return chosen or ["CPUExecutionProvider"]


def _decoder_feed(session, **values: np.ndarray) -> dict[str, np.ndarray]:
    aliases = {
        "image_embeddings": values["embedding"],
        "image_embedding": values["embedding"],
        "point_coords": values["point_coords"],
        "point_labels": values["point_labels"],
        "mask_input": values["mask_input"],
        "has_mask_input": values["has_mask_input"],
        "orig_im_size": values["orig_im_size"],
    }
    feed: dict[str, np.ndarray] = {}
    for meta in session.get_inputs():
        if meta.name not in aliases:
            raise ValueError(f"unsupported MobileSAM decoder input {meta.name}")
        array = aliases[meta.name]
        feed[meta.name] = _fit_input(array, meta)
    return feed


def _fit_input(array: np.ndarray, meta) -> np.ndarray:
    arr = np.asarray(array)
    shape = list(meta.shape)
    while arr.ndim < len(shape):
        arr = arr[None, ...]
    type_name = str(getattr(meta, "type", ""))
    if "int64" in type_name:
        return np.ascontiguousarray(arr.astype(np.int64))
    return np.ascontiguousarray(arr.astype(np.float32))


def _cv2():
    import cv2

    return cv2
