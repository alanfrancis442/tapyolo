import numpy as np

from tapyolo.capture import process_frames, select_negative_indices
from tapyolo.sampler import sample_keyframe_indices, stride_indices
from tapyolo.ssim import structural_similarity


def test_stride_six_indices():
    assert stride_indices(90, 6) == list(range(0, 90, 6))


def test_ssim_identical_and_opposite():
    image = np.zeros((32, 32, 3), np.uint8)
    assert structural_similarity(image, image.copy()) > 0.99
    white = np.full((32, 32, 3), 255, np.uint8)
    assert structural_similarity(image, white) < 0.88


def test_identical_frames_use_stride_and_stay_in_bounds():
    frame = np.zeros((24, 24, 3), np.uint8)
    frames = [frame.copy() for _ in range(120)]
    result = sample_keyframe_indices(frames)
    assert result.method == "stride6"
    assert result.indices == list(range(0, 120, 6))
    assert 20 <= len(result.indices) <= 30


def test_short_static_clip_is_resampled_into_bounds():
    frame = np.full((16, 16, 3), 4, np.uint8)
    frames = [frame.copy() for _ in range(60)]
    result = sample_keyframe_indices(frames)
    assert result.method == "stride6_bounded"
    assert len(result.indices) == 20
    assert result.indices[0] == 0
    assert result.indices[-1] == 59


def test_diverse_frames_stay_inside_the_band():
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8) for _ in range(25)]
    result = sample_keyframe_indices(frames)
    assert result.method == "ssim"
    assert result.indices == list(range(25))

    many = [rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8) for _ in range(80)]
    capped = sample_keyframe_indices(many)
    assert capped.method == "ssim_capped"
    assert len(capped.indices) == 30
    assert capped.indices[0] == 0
    assert capped.indices[-1] == 79


def test_too_few_frames_are_not_padded():
    frames = [np.full((12, 12, 3), index, np.uint8) for index in range(5)]
    result = sample_keyframe_indices(frames)
    assert len(result.indices) <= 5
    assert result.indices == sorted(set(result.indices))


def test_negatives_come_from_the_last_second():
    indices = select_negative_indices(90, fps=30.0)
    assert 3 <= len(indices) <= 5
    assert indices == sorted(set(indices))
    assert min(indices) >= 60


def test_process_frames_separates_keyframes_and_negatives():
    frames = [np.full((32, 32, 3), 7, np.uint8) for _ in range(90)]
    bundle = process_frames(frames, fps=30)
    assert 20 <= len(bundle.keyframes) <= 30
    assert 3 <= len(bundle.negatives) <= 5
    assert set(bundle.keyframe_indices).isdisjoint(bundle.negative_indices)
    assert min(bundle.negative_indices) >= 60
    assert bundle.duration_s == 3.0
    assert bundle.warning is None

    short = process_frames([np.zeros((8, 8, 3), np.uint8) for _ in range(30)], fps=30)
    assert short.warning
    assert "3-8" in short.warning
