"""Propagate the selected object with SAM 2 video memory, in both directions."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tapyolo.bbox import mask_to_bbox


@dataclass
class VideoTrackResult:
    boxes: list
    masks: list
    scores: list[float]
    backend: str = "sam2-video"


def write_video(path: Path, frames, fps=30.0):
    import cv2

    height, width = frames[0].shape[:2]
    # OpenCV's video encoders require even dimensions.
    size = (max(2, width // 2 * 2), max(2, height // 2 * 2))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("Could not prepare the tracking video")
    try:
        for frame in frames:
            image = cv2.resize(frame, size) if frame.shape[1::-1] != size else frame
            writer.write(image)
    finally:
        writer.release()
    return size


def make_predictor(weights, device):
    from ultralytics.models.sam import SAM2VideoPredictor

    class ConditionedVideoPredictor(SAM2VideoPredictor):
        """Use later corrections as conditioning frames for the same object."""

        corrections = None

        def _prepare_prompts(self, *args, **kwargs):
            points, labels, masks = super()._prepare_prompts(*args, **kwargs)
            if masks is not None and masks.ndim == 3:
                masks = masks[:, None]  # SAM2's mask encoder needs N,C,H,W.
            return points, labels, masks

        def inference(self, im, bboxes=None, points=None, labels=None, masks=None):
            frame_number = self.dataset.frame  # Ultralytics video frames are 1-based.
            if frame_number > 1 and frame_number in (self.corrections or {}):
                self.inference_state["im"] = im
                _, _, prepared = self._prepare_prompts(
                    im.shape[2:], self.src_shape, masks=self.corrections[frame_number]
                )
                self.add_new_prompts(0, masks=prepared[[0]], frame_idx=frame_number)
            return super().inference(im, bboxes, points, labels, masks)

    return ConditionedVideoPredictor(
        overrides={
            "model": str(weights),
            "device": device,
            "imgsz": 1024,
            "verbose": False,
            "save": False,
            "batch": 1,
        }
    )


def checked_mask(raw, shape, seed_area, previous=None, gap=1):
    """Reject empty, scene-sized or discontinuous masks, without inventing boxes."""
    import cv2

    height, width = shape
    mask = cv2.resize(
        (np.asarray(raw) > 0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    area = int(mask.sum())
    if (
        area < 9
        or area > 0.85 * mask.size
        or not 0.05 <= area / max(seed_area, 1) <= 10
    ):
        return None
    if previous is not None:
        py, px = np.nonzero(previous)
        ys, xs = np.nonzero(mask)
        distance = np.hypot(xs.mean() - px.mean(), ys.mean() - py.mean())
        if distance > np.hypot(width, height) * 0.25 * min(gap, 4):
            return None
    return mask


class Sam2VideoTracker:
    name = "sam2-video"

    def __init__(self, weights, device="cpu", predictor_factory=None):
        self.weights = Path(weights)
        self.device = device
        self.predictor_factory = predictor_factory or make_predictor
        self._predictor = None

    def propagate(
        self,
        frames,
        start_index,
        seed_mask,
        keyframe_indices,
        *,
        corrections=None,
        excluded=None,
        progress=None,
    ):
        import cv2
        import torch

        if not frames or not 0 <= start_index < len(frames):
            raise ValueError("Invalid tracking sequence or starting frame")
        if any(index < 0 or index >= len(frames) for index in keyframe_indices):
            raise ValueError("Keyframe indices do not map to the tracking sequence")
        shape = seed_mask.shape[:2]
        if int(seed_mask.sum()) < 9:
            raise ValueError("Select a visible object before tracking")
        lookup = {frame_id: index for index, frame_id in enumerate(keyframe_indices)}
        masks = [None] * len(keyframe_indices)
        excluded = set(excluded or ())
        corrections = dict(corrections or {})
        corrections[start_index] = seed_mask
        total = len(frames) + 1  # seed is visited once per direction
        done = 0
        if self._predictor is None:
            self._predictor = self.predictor_factory(self.weights, self.device)
        predictor = self._predictor
        try:
            with (
                tempfile.TemporaryDirectory(prefix="tapyolo-track-") as temporary,
                torch.inference_mode(),
            ):
                for direction in (1, -1):
                    order = (
                        list(range(start_index, len(frames)))
                        if direction == 1
                        else list(range(start_index, -1, -1))
                    )
                    if len(order) == 1:
                        done += 1
                        continue
                    path = Path(temporary) / f"direction_{direction}.avi"
                    size = write_video(path, [frames[index] for index in order])
                    prompts = {
                        offset + 1: cv2.resize(
                            corrections[index].astype(np.uint8),
                            size,
                            interpolation=cv2.INTER_NEAREST,
                        )
                        for offset, index in enumerate(order)
                        if index in corrections
                    }
                    predictor.reset_image()
                    predictor.corrections = prompts
                    stream = predictor(source=str(path), stream=True, masks=prompts[1])
                    previous, gap = seed_mask, 1
                    count = 0
                    try:
                        for offset, result in enumerate(stream):
                            if offset >= len(order):
                                raise RuntimeError(
                                    "Tracker returned too many video frames"
                                )
                            count += 1
                            original = order[offset]
                            mask = None
                            if result.masks is not None and len(result.masks.data):
                                mask = checked_mask(
                                    result.masks.data[0].cpu().numpy(),
                                    shape,
                                    int(seed_mask.sum()),
                                    previous,
                                    gap,
                                )
                            if original in corrections:
                                mask = corrections[original].copy()
                            if original in excluded:
                                mask = None
                            if mask is not None:
                                previous, gap = mask, 1
                            else:
                                gap += 1
                            if original in lookup:
                                masks[lookup[original]] = mask
                            done += 1
                            if progress is not None:
                                progress(done, total)
                    finally:
                        stream.close()
                    if count != len(order):
                        raise RuntimeError(
                            "Tracking video ended before all frames were processed"
                        )
        finally:
            # Do not retain one clip's GPU memory or conditioning in another session.
            predictor.inference_state = {}
            predictor.corrections = {}
            predictor.reset_image()
        for frame_id, mask in corrections.items():
            if frame_id in lookup and frame_id not in excluded:
                masks[lookup[frame_id]] = mask.copy()
        boxes = [None if mask is None else mask_to_bbox(mask) for mask in masks]
        if progress is not None:
            progress(total, total)
        # SAM2VideoPredictor exposes no calibrated per-frame confidence score.
        return VideoTrackResult(
            boxes, masks, [float(mask is not None) for mask in masks]
        )


def load_video_tracker(device="cpu"):
    configured = os.environ.get("TAPYOLO_SAM2_WEIGHTS")
    weights = Path(configured) if configured else Path.cwd() / "weights/sam2.1_t.pt"
    return Sam2VideoTracker(weights, device) if weights.is_file() else None
