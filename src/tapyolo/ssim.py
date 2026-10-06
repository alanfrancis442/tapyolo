"""Structural similarity used to drop near-duplicate video frames."""

from __future__ import annotations

import numpy as np

_C1 = (0.01 * 255.0) ** 2
_C2 = (0.03 * 255.0) ** 2


def structural_similarity(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Windowed SSIM in ``[0, 1]`` (values can sit slightly outside from rounding).

    Frames are converted to grayscale and, when larger than 160 px on a side,
    downscaled so a long clip stays cheap to filter.
    """
    gray_a = _prepare(image_a)
    gray_b = _prepare(image_b)
    if gray_a.shape != gray_b.shape:
        raise ValueError("frames must share a shape for SSIM")
    k = min(11, gray_a.shape[0], gray_a.shape[1])
    if k % 2 == 0:
        k -= 1
    if k < 3:
        score = _global_ssim(gray_a, gray_b)
    else:
        score = _window_ssim(gray_a, gray_b, k)
    if not np.isfinite(score):
        return 0.0
    return float(score)


def _prepare(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr.astype(np.float64).mean(axis=2)
    else:
        arr = arr.astype(np.float64)
    if np.issubdtype(image.dtype, np.floating) and arr.size and float(arr.max()) <= 1.0:
        arr = arr * 255.0
    return _downscale(arr, max_side=160)


def _downscale(gray: np.ndarray, max_side: int) -> np.ndarray:
    height, width = gray.shape
    long_side = max(height, width)
    if long_side <= max_side:
        return gray
    scale = max_side / float(long_side)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    ys = np.linspace(0, height - 1, new_h).astype(np.int32)
    xs = np.linspace(0, width - 1, new_w).astype(np.int32)
    return gray[ys][:, xs]


def _global_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    mu_a = float(gray_a.mean())
    mu_b = float(gray_b.mean())
    var_a = float(gray_a.var())
    var_b = float(gray_b.var())
    cov = float(((gray_a - mu_a) * (gray_b - mu_b)).mean())
    numerator = (2 * mu_a * mu_b + _C1) * (2 * cov + _C2)
    denominator = (mu_a**2 + mu_b**2 + _C1) * (var_a + var_b + _C2)
    return numerator / denominator


def _window_ssim(gray_a: np.ndarray, gray_b: np.ndarray, k: int) -> float:
    mu_a = _box_blur(gray_a, k)
    mu_b = _box_blur(gray_b, k)
    sigma_a = np.maximum(_box_blur(gray_a * gray_a, k) - mu_a * mu_a, 0.0)
    sigma_b = np.maximum(_box_blur(gray_b * gray_b, k) - mu_b * mu_b, 0.0)
    sigma_ab = _box_blur(gray_a * gray_b, k) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + _C1) * (2 * sigma_ab + _C2)
    denominator = (mu_a * mu_a + mu_b * mu_b + _C1) * (sigma_a + sigma_b + _C2)
    return float((numerator / denominator).mean())


def _box_blur(image: np.ndarray, k: int) -> np.ndarray:
    radius = k // 2
    padded = np.pad(image, radius, mode="reflect")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    height, width = image.shape
    y0 = np.arange(height)[:, None]
    x0 = np.arange(width)[None, :]
    y1 = y0 + k
    x1 = x0 + k
    total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return total / float(k * k)
