"""Unit/API tests never depend on local learned weights or GPU availability."""

import pytest

from tapyolo.segment import FallbackSegmenter, SegmenterChoice


@pytest.fixture(autouse=True)
def deterministic_app_segmenter(monkeypatch):
    monkeypatch.setattr(
        "tapyolo.ui.load_segmenter",
        lambda **kwargs: SegmenterChoice(FallbackSegmenter(), "test fallback"),
    )

    monkeypatch.setattr("tapyolo.ui.load_video_tracker", lambda **kwargs: None)
