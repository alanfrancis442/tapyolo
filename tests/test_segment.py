import numpy as np

from tapyolo.bbox import mask_to_bbox
from tapyolo.segment import (
    FallbackSegmenter,
    load_segmenter,
    prepare_encoder_input,
    scale_point,
)


def test_fallback_covers_a_solid_blob():
    image = np.full((80, 100, 3), 10, np.uint8)
    image[20:50, 30:70] = (0, 0, 255)
    mask, detail = FallbackSegmenter().segment(image, 50, 35)
    assert detail == "flood"
    assert mask_to_bbox(mask) == (30, 20, 69, 49)


def test_segmenter_falls_back_without_weights(tmp_path, monkeypatch):
    monkeypatch.delenv("TAPYOLO_MOBILESAM_ENCODER", raising=False)
    monkeypatch.delenv("TAPYOLO_MOBILESAM_DECODER", raising=False)
    choice = load_segmenter([tmp_path])
    assert choice.name == "fallback"
    assert choice.onnx_error == "weights not found"


def test_encoder_preprocess_shape_and_point_scale():
    image = np.zeros((100, 200, 3), np.uint8)
    tensor, scale = prepare_encoder_input(image, 1024)
    assert tensor.shape == (1, 3, 1024, 1024)
    assert tensor.dtype == np.float32
    assert scale_point(50, 25, scale) == (50 * scale, 25 * scale)
