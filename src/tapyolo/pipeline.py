"""Connect capture, a tap, and bidirectional tracks."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from tapyolo.bbox import xyxy_to_yolo
from tapyolo.capture import CaptureBundle, process_frames, read_video
from tapyolo.segment import SegmentOutput, segment_point
from tapyolo.tracker import propagate_bidirectional


def load_video(path: Path) -> CaptureBundle:
    frames, fps = read_video(path)
    return process_frames(frames, fps)


def tap_keyframes(
    frames: list[np.ndarray],
    index: int,
    x: int,
    y: int,
    segmenter,
    *,
    video_tracker=None,
    tracking_frames=None,
    keyframe_indices=None,
) -> dict:
    if not frames:
        raise ValueError("no keyframes to tap")
    if not 0 <= index < len(frames):
        raise ValueError("keyframe index is out of range")
    output: SegmentOutput = segment_point(frames[index], x, y, segmenter)
    if output.bbox is None:
        raise ValueError("segmenter returned an empty mask")
    started = time.perf_counter()
    if video_tracker is not None:
        dense = tracking_frames if tracking_frames else frames
        mapping = keyframe_indices if tracking_frames else list(range(len(frames)))
        propagated = video_tracker.propagate(
            dense, mapping[index], output.mask, mapping
        )
    else:
        propagated = propagate_bidirectional(frames, index, output.bbox)
    track_ms = (time.perf_counter() - started) * 1000.0
    height, width = frames[index].shape[:2]
    tracks = []
    for frame_index, box in enumerate(propagated.boxes):
        yolo = (
            None
            if box is None
            else list(xyxy_to_yolo(box[0], box[1], box[2], box[3], width, height))
        )
        tracks.append(
            {
                "index": frame_index,
                "bbox": None if box is None else list(box),
                "yolo": yolo,
                "score": propagated.scores[frame_index],
            }
        )
    return {
        "backend": output.backend,
        "detail": output.detail,
        "elapsed_ms": output.elapsed_ms,
        "track_ms": track_ms,
        "bbox": list(output.bbox),
        "yolo": None if output.yolo is None else list(output.yolo),
        "tracks": tracks,
        "tracker": propagated.backend,
        "mask": output.mask,
    }
