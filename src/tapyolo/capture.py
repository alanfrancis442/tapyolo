"""Video capture, keyframe export, and hard-negative tail extraction."""

from __future__ import annotations

import math
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tapyolo.sampler import evenly_spaced, sample_keyframe_indices

TAIL_SECONDS = 1.0
NEGATIVE_COUNT = 4
TARGET_MIN_SECONDS = 3.0
TARGET_MAX_SECONDS = 8.0
FRAME_BUNDLE_MAGIC = b"TYFR"
MAX_BUNDLE_FRAMES = 300


@dataclass
class CaptureBundle:
    fps: float
    width: int
    height: int
    duration_s: float
    keyframe_indices: list[int]
    keyframes: list[np.ndarray]
    negative_indices: list[int]
    negatives: list[np.ndarray]
    sample_method: str
    warning: str | None
    tracking_frames: list[np.ndarray] = field(default_factory=list)


@dataclass
class SavedCapture:
    root: Path
    keyframe_paths: list[Path]
    negative_paths: list[Path]
    negative_label_paths: list[Path]


def sanitize_fps(fps: float) -> float:
    try:
        value = float(fps)
    except (TypeError, ValueError):
        return 30.0
    if not math.isfinite(value) or value < 1.0 or value > 120.0:
        return 30.0
    return value


def select_negative_indices(
    n_frames: int,
    fps: float,
    *,
    tail_seconds: float = TAIL_SECONDS,
    count: int = NEGATIVE_COUNT,
) -> list[int]:
    """Evenly pick up to ``count`` frames from the last ``tail_seconds``."""
    if n_frames <= 0:
        return []
    safe_fps = sanitize_fps(fps)
    tail = max(1, int(round(safe_fps * tail_seconds)))
    tail = min(tail, n_frames)
    pool = list(range(n_frames - tail, n_frames))
    take = min(max(1, count), len(pool))
    return evenly_spaced(pool, take)


def content_end(n_frames: int, fps: float, tail_seconds: float = TAIL_SECONDS) -> int:
    """Index where the negative tail starts. Keyframes are sampled before it."""
    if n_frames <= 0:
        return 0
    safe_fps = sanitize_fps(fps)
    tail = max(1, int(round(safe_fps * tail_seconds)))
    tail = min(tail, n_frames)
    return n_frames - tail


def process_frames(frames: list[np.ndarray], fps: float) -> CaptureBundle:
    if not frames:
        raise ValueError("video has no frames")
    safe_fps = sanitize_fps(fps)
    height, width = frames[0].shape[:2]
    for frame in frames:
        if frame.shape[0] != height or frame.shape[1] != width:
            raise ValueError("frames differ in size")
    duration = len(frames) / safe_fps
    warning = None
    if not (TARGET_MIN_SECONDS <= duration <= TARGET_MAX_SECONDS):
        warning = (
            f"Video duration is {duration:.2f}s. Capture is designed for "
            f"{TARGET_MIN_SECONDS:.0f}-{TARGET_MAX_SECONDS:.0f} seconds."
        )
    end = content_end(len(frames), safe_fps)
    negative_idx = select_negative_indices(len(frames), safe_fps)
    content_ids = list(range(end))
    content_frames = [frames[i] for i in content_ids]
    sampled = sample_keyframe_indices(content_frames)
    key_idx = [content_ids[i] for i in sampled.indices] if content_ids else []
    return CaptureBundle(
        fps=safe_fps,
        width=width,
        height=height,
        duration_s=duration,
        keyframe_indices=key_idx,
        keyframes=[frames[i] for i in key_idx],
        negative_indices=negative_idx,
        negatives=[frames[i] for i in negative_idx],
        sample_method=sampled.method,
        warning=warning,
        tracking_frames=tracking_frames(content_frames),
    )


def tracking_frames(frames, max_side=640):
    """Keep consecutive frames for video memory, bounded in resolution."""
    cv2 = _cv2()
    if not frames:
        return []
    height, width = frames[0].shape[:2]
    scale = min(1.0, max_side / max(height, width))
    size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
    return [
        cv2.resize(frame, size) if frame.shape[1::-1] != size else frame
        for frame in frames
    ]


def save_capture(bundle: CaptureBundle, root: Path) -> SavedCapture:
    """Write keyframe JPEGs and empty label files for the negative tail."""
    from tapyolo.labels import write_empty_label

    cv2 = _cv2()
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    key_paths: list[Path] = []
    for index, frame in enumerate(bundle.keyframes):
        path = images / f"key_{index:03d}.jpg"
        if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"failed to write {path}")
        key_paths.append(path)
    neg_paths: list[Path] = []
    neg_labels: list[Path] = []
    for index, frame in enumerate(bundle.negatives):
        path = images / f"neg_{index:03d}.jpg"
        if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"failed to write {path}")
        label = labels / f"neg_{index:03d}.txt"
        write_empty_label(label)
        neg_paths.append(path)
        neg_labels.append(label)
    if bundle.tracking_frames:
        from tapyolo.video_tracker import write_video

        write_video(root / "tracking.avi", bundle.tracking_frames, bundle.fps)
    return SavedCapture(root, key_paths, neg_paths, neg_labels)


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video {path}")
    fps = sanitize_fps(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(_as_bgr(frame))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video has no frames: {path}")
    return frames, fps


def decode_video_bytes(data: bytes) -> tuple[list[np.ndarray], float]:
    if not data:
        raise ValueError("video has no frames")
    suffix = (
        ".avi"
        if data.startswith(b"RIFF")
        else ".webm"
        if data.startswith(b"\x1aE\xdf\xa3")
        else ".mp4"
    )
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return read_video(Path(tmp.name))
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def encode_frame_bundle(frames: list[np.ndarray], fps: float) -> bytes:
    cv2 = _cv2()
    parts: list[bytes] = [
        FRAME_BUNDLE_MAGIC,
        struct.pack(">II", len(frames), int(round(sanitize_fps(fps) * 1000))),
    ]
    for frame in frames:
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("png encode failed")
        blob = encoded.tobytes()
        parts.append(struct.pack(">I", len(blob)))
        parts.append(blob)
    return b"".join(parts)


def decode_frame_bundle(data: bytes) -> tuple[list[np.ndarray], float]:
    if len(data) < 12 or data[:4] != FRAME_BUNDLE_MAGIC:
        raise ValueError("frame bundle header is invalid")
    count, fps_milli = struct.unpack_from(">II", data, 4)
    if count > MAX_BUNDLE_FRAMES:
        raise ValueError(f"bundle exceeds {MAX_BUNDLE_FRAMES} frames")
    fps = sanitize_fps(fps_milli / 1000.0)
    cv2 = _cv2()
    offset = 12
    frames: list[np.ndarray] = []
    for _ in range(count):
        if offset + 4 > len(data):
            raise ValueError("frame bundle ended early")
        (size,) = struct.unpack_from(">I", data, offset)
        offset += 4
        blob = data[offset : offset + size]
        offset += size
        if len(blob) != size:
            raise ValueError("frame bundle ended early")
        image = cv2.imdecode(
            np.frombuffer(blob, dtype=np.uint8).copy(), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("could not decode a bundled frame")
        frames.append(image)
    if not frames:
        raise ValueError("video has no frames")
    return frames, fps


def decode_upload(data: bytes, content_type: str) -> tuple[list[np.ndarray], float]:
    if "application/x-tapyolo-frames" in content_type:
        return decode_frame_bundle(data)
    return decode_video_bytes(data)


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _cv2():
    import cv2

    return cv2
