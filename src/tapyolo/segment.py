"""Point-prompt segmentation: MobileSAM ONNX when weights exist, else flood fill / GrabCut."""

from __future__ import annotations

import hashlib
import os
import threading
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
        raise ValueError(
            "Could not isolate the object. Add a foreground/background point or draw a correction box."
        )

    def segment_prompt(self, image, points, labels):
        positives = [point for point, label in zip(points, labels) if label == 1]
        mask = np.zeros(image.shape[:2], np.uint8)
        for x, y in positives:
            candidate, _ = self.segment(image, x, y)
            mask |= candidate
        for (x, y), label in zip(points, labels):
            if label == 0 and mask[y, x]:
                mask[_component_at(mask, x, y) > 0] = 0
        return validate_prompt_mask(
            mask, points, labels
        ), "OpenCV fallback; review the selection"


class MobileSamSegmenter:
    """Learned point prompts with one cached image embedding for refinement."""

    name = "mobilesam"

    def __init__(self, weights: Path, device: str = "cpu") -> None:
        from ultralytics.models.sam import Predictor

        self.predictor = Predictor(
            overrides={
                "model": str(weights),
                "device": device,
                "imgsz": 1024,
                "verbose": False,
                "save": False,
            }
        )
        self._key = None
        self._lock = threading.Lock()

    def segment(self, image, x, y):
        return self.segment_prompt(image, [[x, y]], [1])

    def segment_prompt(self, image, points, labels):
        import torch

        key = (image.shape, hashlib.blake2b(image.tobytes(), digest_size=16).digest())
        with self._lock, torch.inference_mode():
            if key != self._key:
                self.predictor.reset_image()
                self.predictor.set_image(image)
                self._key = key
            results = self.predictor(
                source=image,
                points=[points],
                labels=[labels],
                multimask_output=True,
            )
            result = results[0]
            if result.masks is None:
                raise ValueError(
                    "MobileSAM returned no object; refine the points or draw a box"
                )
            masks = result.masks.data.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            return choose_prompt_mask(
                masks, scores, points, labels
            ), "MobileSAM with point refinement"


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
        self.encoder = ort.InferenceSession(
            str(encoder_path), sess_options=options, providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(decoder_path), sess_options=options, providers=providers
        )
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self._embedding_key = None
        self._embedding = None

    def segment(self, image: np.ndarray, x: int, y: int) -> tuple[np.ndarray, str]:
        return self.segment_prompt(image, [[x, y]], [1])

    def segment_prompt(self, image, points, labels):
        cv2 = _cv2()
        frame = image
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        height, width = frame.shape[:2]
        input_size = _spatial_size(self.encoder)
        tensor, scale = prepare_encoder_input(frame, input_size)
        enc_input = self.encoder.get_inputs()[0]
        key = (frame.shape, hashlib.blake2b(frame.tobytes(), digest_size=16).digest())
        if key != self._embedding_key:
            self._embedding = self.encoder.run(None, {enc_input.name: tensor})[0]
            self._embedding_key = key
        embedding = self._embedding
        point = np.array(
            [[*[[x * scale, y * scale] for x, y in points], [0, 0]]], dtype=np.float32
        )
        labels_tensor = np.array([[*labels, -1]], dtype=np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask = np.zeros((1,), dtype=np.float32)
        orig = np.array([height, width], dtype=np.float32)
        feed = _decoder_feed(
            self.decoder,
            embedding=embedding,
            point_coords=point,
            point_labels=labels_tensor,
            mask_input=mask_input,
            has_mask_input=has_mask,
            orig_im_size=orig,
        )
        outputs = self.decoder.run(None, feed)
        raw = np.asarray(outputs[0]).reshape(-1, *outputs[0].shape[-2:])
        candidates = [
            postprocess_sam_mask(item, None, (height, width), input_size)
            for item in raw
        ]
        scores = (
            outputs[1].reshape(-1) if len(outputs) > 1 else np.ones(len(candidates))
        )
        mask = choose_prompt_mask(candidates, scores, points, labels)
        return mask, "onnx"


@dataclass
class SegmenterChoice:
    segmenter: FallbackSegmenter | OnnxMobileSamSegmenter | MobileSamSegmenter
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


def prepare_encoder_input(
    image: np.ndarray, input_size: int = SAM_INPUT
) -> tuple[np.ndarray, float]:
    """RGB, longest-side resize, pad to square, SAM mean/std. Returns NCHW and the scale."""
    cv2 = _cv2()
    rgb = (
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.ndim == 3
        else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    )
    height, width = rgb.shape[:2]
    scale = float(input_size) / float(max(height, width))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((input_size, input_size, 3), dtype=np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    canvas[:resized_h, :resized_w] = (resized.astype(np.float32) - mean) / std
    tensor = np.transpose(canvas, (2, 0, 1))[None].astype(np.float32)
    return tensor, scale


def scale_point(x: float, y: float, scale: float) -> tuple[float, float]:
    return x * scale, y * scale


def postprocess_sam_mask(
    masks: np.ndarray,
    ious: np.ndarray | None,
    original_hw: tuple[int, int],
    input_size: int = SAM_INPUT,
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
    height, width = original_hw
    if arr.shape != (height, width):
        cv2 = _cv2()
        # Low-resolution decoder logits include the square encoder's padding.
        arr = cv2.resize(arr.astype(np.float32), (input_size, input_size))
        scale = input_size / max(height, width)
        arr = arr[: max(1, round(height * scale)), : max(1, round(width * scale))]
        arr = cv2.resize(arr, (width, height))
    return (arr > 0).astype(np.uint8)


def validate_prompt_mask(mask, points, labels):
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.sum() < 9 or binary.mean() > 0.85:
        raise ValueError(
            "Selection is empty or covers most of the scene. Refine the points or draw a box."
        )
    for (x, y), label in zip(points, labels):
        if bool(binary[y, x]) != bool(label):
            raise ValueError(
                "Selection does not match the points. Refine the points or draw a box."
            )
    return binary


def choose_prompt_mask(masks, scores, points, labels):
    for index in np.argsort(np.asarray(scores).reshape(-1))[::-1]:
        try:
            return validate_prompt_mask(masks[index], points, labels)
        except ValueError:
            continue
    raise ValueError(
        "No plausible mask matches the points. Add correction points or draw a box."
    )


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


def load_segmenter(
    roots: list[Path] | None = None, device: str = "cpu"
) -> SegmenterChoice:
    found = find_onnx_weights(roots)
    if found is None:
        native = os.environ.get("TAPYOLO_MOBILESAM_WEIGHTS")
        candidates = (
            [Path(native)]
            if native
            else [root / "mobile_sam.pt" for root in (roots or default_weight_roots())]
        )
        for path in candidates:
            if path.is_file():
                try:
                    return SegmenterChoice(
                        MobileSamSegmenter(path, device),
                        onnx_error="ONNX pair not installed; using PyTorch MobileSAM",
                    )
                except Exception as exc:
                    return SegmenterChoice(FallbackSegmenter(), onnx_error=str(exc))
        return SegmenterChoice(FallbackSegmenter(), onnx_error="weights not found")
    try:
        return SegmenterChoice(
            OnnxMobileSamSegmenter(found[0], found[1]), onnx_error=None
        )
    except Exception as exc:
        return SegmenterChoice(FallbackSegmenter(), onnx_error=str(exc))


def segment_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Separate the object inside a drawn rectangle from the wall and the hand.

    The color along the box edge is treated as background. Skin-colored pixels
    are treated as background too, so a loose box does not keep the whole room.
    """
    cv2 = _cv2()
    frame = np.ascontiguousarray(image)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, width, height)
    mask = np.zeros((height, width), np.uint8)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return mask
    roi = frame[y1 : y2 + 1, x1 : x2 + 1]
    grab = _mask_grabcut(roi)
    cluster = _mask_cluster(roi)
    chosen = (
        grab if _mask_quality(roi, grab) >= _mask_quality(roi, cluster) else cluster
    )
    chosen = _grow_into_object(roi, chosen)
    box_area = roi.shape[0] * roi.shape[1]
    if int(chosen.sum()) < max(16, int(0.02 * box_area)):
        chosen = np.ones(roi.shape[:2], np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = (chosen > 0).astype(np.uint8)
    return mask


def object_mask(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Mask used for training.

    A segmented object is kept when it covers a real part of the box. A tiny
    fragment falls back to the whole box so the pale body of the object is not
    thrown away.
    """
    mask = segment_box(image, box)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, width, height)
    area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
    if int(mask.sum()) < 0.12 * area:
        full = np.zeros((height, width), np.uint8)
        full[y1 : y2 + 1, x1 : x2 + 1] = 1
        return full
    return mask


def scene_without_object(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The same frame with the segmented object painted out.

    Training treats this as background, so the room and the person are not enough
    on their own to count as the object.
    """
    cv2 = _cv2()
    frame = np.ascontiguousarray(image)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    hole = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if hole.shape[:2] != frame.shape[:2] or int(hole.sum()) < 16:
        return frame.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hole = cv2.dilate(hole, kernel)
    return cv2.inpaint(frame, hole, 3, cv2.INPAINT_TELEA)


def training_views(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    mask: np.ndarray,
    *,
    copies: int = 5,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Real frame first, then the same object on other backgrounds."""
    frame = np.ascontiguousarray(image)
    views = [(frame, tuple(int(value) for value in box))]
    views.extend(cutout_views(frame, mask, copies=copies, rng=rng))
    return views


def cutout_views(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    copies: int = 5,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Paste the segmented object onto new backgrounds.

    The first view keeps the object's place in the frame. The rest move and
    scale it, so the only pixels shared by every sample are the object itself.
    """
    cv2 = _cv2()
    frame = np.ascontiguousarray(image)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.shape[:2] != frame.shape[:2]:
        raise ValueError("mask and image sizes differ")
    bounds = mask_to_bbox(binary)
    if bounds is None:
        return []
    if rng is None:
        rng = np.random.default_rng(0)
    x1, y1, x2, y2 = bounds
    sprite = frame[y1 : y2 + 1, x1 : x2 + 1]
    alpha = binary[y1 : y2 + 1, x1 : x2 + 1].astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    height, width = frame.shape[:2]
    views: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []

    def place(scale: float, left: int, top: int) -> None:
        nh = max(4, int(round(sprite.shape[0] * scale)))
        nw = max(4, int(round(sprite.shape[1] * scale)))
        resized = cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_LINEAR)
        scaled = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = _random_background(height, width, rng)
        tight = _composite(canvas, resized, scaled, left, top)
        if tight is None:
            return
        if tight[2] - tight[0] < 4 or tight[3] - tight[1] < 4:
            return
        views.append((canvas, tight))

    place(1.0, x1, y1)
    for _ in range(max(0, copies)):
        scale = float(rng.uniform(0.75, 1.2))
        nh = max(4, int(round(sprite.shape[0] * scale)))
        nw = max(4, int(round(sprite.shape[1] * scale)))
        max_left = max(0, width - nw)
        max_top = max(0, height - nh)
        place(
            scale, int(rng.integers(0, max_left + 1)), int(rng.integers(0, max_top + 1))
        )
    return views


def segment_point(
    image: np.ndarray,
    x: int,
    y: int,
    segmenter: FallbackSegmenter | OnnxMobileSamSegmenter,
) -> SegmentOutput:
    """Time a validated selection; learned-model failures require user correction."""
    started = time.perf_counter()
    mask, detail = segmenter.segment(image, x, y)
    x, y = _clip_point(x, y, image.shape[1], image.shape[0])
    mask = validate_prompt_mask(mask, [[x, y]], [1])
    backend = segmenter.name
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    bbox = mask_to_bbox(mask)
    height, width = image.shape[:2]
    yolo = None if bbox is None else xyxy_to_yolo(*bbox, width, height)
    return SegmentOutput(mask, bbox, yolo, elapsed_ms, backend, detail)


def _clip_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return int(min(max(0, x), width - 1)), int(min(max(0, y), height - 1))


def _clip_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(value) for value in box)
    left, right = sorted(
        (_clip_point(x1, 0, width, height)[0], _clip_point(x2, 0, width, height)[0])
    )
    top, bottom = sorted(
        (_clip_point(0, y1, width, height)[1], _clip_point(0, y2, width, height)[1])
    )
    return left, top, right, bottom


def _downscale(roi: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    cv2 = _cv2()
    height, width = roi.shape[:2]
    long_side = max(height, width)
    scale = 1.0 if long_side <= 280 else 280 / long_side
    resized_w = max(8, int(round(width * scale)))
    resized_h = max(8, int(round(height * scale)))
    if (resized_w, resized_h) == (width, height):
        return roi, (width, height)
    return cv2.resize(roi, (resized_w, resized_h), interpolation=cv2.INTER_AREA), (
        width,
        height,
    )


def _edge_band(height: int, width: int) -> np.ndarray:
    band = max(2, min(height, width) // 12)
    border = np.zeros((height, width), np.uint8)
    border[:band, :] = 1
    border[-band:, :] = 1
    border[:, :band] = 1
    border[:, -band:] = 1
    return border


def _skin_mask(image: np.ndarray) -> np.ndarray:
    """Skin-toned pixels, leaving vivid object colors in the foreground."""
    cv2 = _cv2()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hue, sat, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    reddish = (hue <= 35) | (hue >= 165)
    classic = (
        (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (sat >= 28) & reddish
    )
    dark = reddish & (sat >= 30) & (value <= 110) & (sat <= 160)
    vivid = sat >= 140
    return (classic | dark) & (~vivid)


def _wall_distance(image: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = _edge_band(*image.shape[:2])
    background = np.median(lab[border > 0].reshape(-1, 3), axis=0)
    return np.linalg.norm(lab - background, axis=2)


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    flood = binary.copy()
    holes = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, holes, (0, 0), 2)
    return np.where(flood == 0, 1, binary).astype(np.uint8)


def _keep_component(binary: np.ndarray, seed: np.ndarray | None) -> np.ndarray:
    cv2 = _cv2()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    best = 1
    best_score = -1
    for index in range(1, count):
        if seed is None:
            score = int(stats[index, cv2.CC_STAT_AREA])
        else:
            score = int(((labels == index) & (seed > 0)).sum())
        if score > best_score:
            best_score = score
            best = index
    return (labels == best).astype(np.uint8)


def _mask_grabcut(roi: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    small, size = _downscale(roi)
    height, width = small.shape[:2]
    skin = _skin_mask(small)
    border = _edge_band(height, width)
    distance = _wall_distance(small)
    wall = distance < 16
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    colorful = (sat >= 50) & (~skin) & (~wall)
    labels = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
    labels[skin | wall | (border > 0)] = cv2.GC_BGD
    labels[colorful] = cv2.GC_FGD
    labels[(~skin) & (~wall) & (distance >= 16) & (~colorful)] = cv2.GC_PR_FGD
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            small, labels, None, background, foreground, 3, cv2.GC_INIT_WITH_MASK
        )
        binary = np.where(
            (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)
    except cv2.error:
        binary = colorful.astype(np.uint8)
    binary[skin] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = _keep_component(binary, colorful.astype(np.uint8))
    binary = _fill_holes(binary)
    binary[skin] = 0
    return cv2.resize(binary, size, interpolation=cv2.INTER_NEAREST)


def _mask_cluster(roi: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    small, size = _downscale(roi)
    height, width = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    skin = _skin_mask(small)
    border = _edge_band(height, width)
    distance = _wall_distance(small)
    wall = distance < 18
    candidate = (~wall) & (~skin)
    ys, xs = np.nonzero(candidate)
    foreground = np.zeros((height, width), np.uint8)
    if len(xs) >= 40:
        samples = lab[ys, xs].reshape(-1, 3)
        clusters = 2 if len(xs) >= 80 else 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.8)
        _, packed, _ = cv2.kmeans(
            samples, clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        label_map = np.zeros((height, width), np.int32)
        label_map[ys, xs] = packed.ravel() + 1

        def score(index: int) -> float:
            selected = label_map == index
            area = int(selected.sum())
            if area < 12:
                return -1e9
            saturation = float(hsv[:, :, 1][selected].mean())
            touch = float((selected & (border > 0)).sum()) / area
            yy, xx = np.nonzero(selected)
            spread = float(
                np.sqrt(((yy - yy.mean()) ** 2 + (xx - xx.mean()) ** 2).mean())
                / max(np.sqrt(area), 1)
            )
            return saturation - 140 * touch - 20 * spread

        best = max(range(1, clusters + 1), key=score)
        foreground = (label_map == best).astype(np.uint8)
    if int(foreground.sum()) < 16:
        foreground = candidate.astype(np.uint8)
    radius = max(2, int(round(0.4 * np.sqrt(max(int(foreground.sum()), 1)))))
    radius = min(radius, max(2, min(height, width) // 6))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    grown = cv2.dilate(foreground, kernel)
    grown[skin] = 0
    if int(foreground.sum()):
        seed = np.median(lab[foreground > 0].reshape(-1, 3), axis=0)
        near = np.linalg.norm(lab - seed, axis=2) < 38
        grown = ((grown > 0) & (near | (foreground > 0))).astype(np.uint8)
    grown = _keep_component(grown, foreground)
    grown = cv2.morphologyEx(
        grown, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    grown = _fill_holes(grown)
    grown[skin] = 0
    return cv2.resize(grown, size, interpolation=cv2.INTER_NEAREST)


def _mask_quality(roi: np.ndarray, mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area < 16:
        return -1e6
    fraction = area / float(mask.size)
    cv2 = _cv2()
    sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    saturation = float(sat[mask > 0].mean())
    skin_fraction = float((_skin_mask(roi) & (mask > 0)).sum()) / area
    penalty = 90.0 if fraction < 0.04 or fraction > 0.7 else 0.0
    return saturation - 110 * skin_fraction - penalty


def _grow_into_object(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Include pale parts of the object that sit against the colored seed."""
    cv2 = _cv2()
    binary = (mask > 0).astype(np.uint8)
    if int(binary.sum()) < 16:
        return binary
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    skin = _skin_mask(roi)
    seed = np.median(lab[binary > 0].reshape(-1, 3), axis=0)
    radius = max(2, int(round(0.28 * np.sqrt(float(binary.sum())))))
    radius = min(radius, max(2, min(roi.shape[:2]) // 8))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    grown = cv2.dilate(binary, kernel)
    near = np.linalg.norm(lab - seed, axis=2) < 42
    grown = ((grown > 0) & (near | (binary > 0)) & (~skin)).astype(np.uint8)
    grown = _keep_component(grown, binary)
    grown = _fill_holes(grown)
    grown[skin] = 0
    return grown


def _random_background(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    cv2 = _cv2()
    coarse = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        noise = rng.integers(0, 40, size=(height, width, 1), dtype=np.int16)
        field = np.clip(field.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return field


def _composite(
    canvas: np.ndarray,
    sprite: np.ndarray,
    alpha: np.ndarray,
    left: int,
    top: int,
) -> tuple[int, int, int, int] | None:
    height, width = canvas.shape[:2]
    sprite_h, sprite_w = sprite.shape[:2]
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(width, left + sprite_w), min(height, top + sprite_h)
    if x2 <= x1 or y2 <= y1:
        return None
    sx1, sy1 = x1 - left, y1 - top
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    part = alpha[sy1:sy2, sx1:sx2].astype(np.float32)
    if float(part.max()) < 0.2:
        return None
    region = canvas[y1:y2, x1:x2].astype(np.float32)
    blended = (
        part[:, :, None] * sprite[sy1:sy2, sx1:sx2].astype(np.float32)
        + (1.0 - part[:, :, None]) * region
    )
    canvas[y1:y2, x1:x2] = blended.astype(np.uint8)
    ys, xs = np.nonzero(part > 0.35)
    if xs.size == 0:
        return None
    return (
        x1 + int(xs.min()),
        y1 + int(ys.min()),
        x1 + int(xs.max()),
        y1 + int(ys.max()),
    )


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
    cv2.grabCut(
        image.copy(), mask, (x0, y0, rect_w, rect_h), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT
    )
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(
        np.uint8
    )
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
