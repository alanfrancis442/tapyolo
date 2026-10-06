"""Keyframe thinning: SSIM at 0.88, then a stride-6 fallback, bounded to 20–30."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tapyolo.ssim import structural_similarity

SSIM_THRESHOLD = 0.88
KEYFRAME_MIN = 20
KEYFRAME_MAX = 30
STRIDE = 6


@dataclass(frozen=True)
class SampleResult:
    indices: list[int]
    method: str


def stride_indices(n_frames: int, stride: int = STRIDE) -> list[int]:
    if n_frames <= 0:
        return []
    step = max(1, stride)
    return list(range(0, n_frames, step))


def evenly_spaced(indices: list[int], count: int) -> list[int]:
    """Pick ``count`` items, always keeping the ends when ``count`` allows it."""
    n = len(indices)
    if count >= n:
        return list(indices)
    if count <= 0:
        return []
    used: set[int] = set()
    chosen: list[int] = []
    for i in range(count):
        pos = int(round(i * (n - 1) / (count - 1))) if count > 1 else 0
        if pos in used:
            for delta in range(1, n):
                left, right = pos - delta, pos + delta
                if left >= 0 and left not in used:
                    pos = left
                    break
                if right < n and right not in used:
                    pos = right
                    break
        used.add(pos)
        chosen.append(pos)
    return [indices[pos] for pos in sorted(chosen)]


def greedy_ssim_indices(
    frames: list[np.ndarray] | tuple[np.ndarray, ...],
    threshold: float = SSIM_THRESHOLD,
) -> list[int]:
    """Keep a frame when its SSIM against the last kept frame drops below ``threshold``."""
    if not frames:
        return []
    selected = [0]
    last = frames[0]
    for index in range(1, len(frames)):
        score = structural_similarity(last, frames[index])
        if score < threshold:
            selected.append(index)
            last = frames[index]
    return selected


def sample_keyframe_indices(
    frames: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    threshold: float = SSIM_THRESHOLD,
    stride: int = STRIDE,
    minimum: int = KEYFRAME_MIN,
    maximum: int = KEYFRAME_MAX,
) -> SampleResult:
    """Return 20–30 keyframe indices when the clip has at least ``minimum`` frames.

    SSIM gating runs first. A result shorter than ``minimum`` is replaced by
    every ``stride``-th frame when that sequence is longer. A result longer
    than ``maximum`` is subsampled. When stride-6 is still short and the clip
    has enough frames, indices are spaced evenly across the clip so the
    20–30 acceptance band holds.
    """
    n_frames = len(frames)
    if n_frames == 0:
        return SampleResult([], "empty")

    selected = greedy_ssim_indices(frames, threshold)
    method = "ssim"
    if len(selected) < minimum:
        stepped = stride_indices(n_frames, stride)
        if len(stepped) > len(selected):
            selected = stepped
            method = "stride6"
    if len(selected) > maximum:
        selected = evenly_spaced(selected, maximum)
        method = f"{method}_capped"
    elif len(selected) < minimum and n_frames >= minimum:
        selected = evenly_spaced(list(range(n_frames)), minimum)
        method = "stride6_bounded"
    return SampleResult(selected, method)
