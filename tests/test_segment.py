import numpy as np

from tapyolo.bbox import mask_to_bbox
from tapyolo.segment import (
    FallbackSegmenter,
    cutout_views,
    load_segmenter,
    training_views,
    prepare_encoder_input,
    scale_point,
    segment_box,
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


def test_segment_box_keeps_the_object_and_drops_the_background():
    image = np.full((120, 160, 3), (180, 180, 180), np.uint8)
    image[40:90, 50:110] = (255, 40, 40)
    mask = segment_box(image, (20, 15, 145, 105))
    assert mask[5, 5] == 0
    assert mask[60, 70] == 1
    assert mask[100, 150] == 0
    views = cutout_views(image, mask, copies=2, rng=np.random.default_rng(1))
    assert len(views) == 3
    placed, box = views[0]
    assert placed.shape == image.shape
    assert not np.all(placed[:12, :12] == 180)
    x1, y1, x2, y2 = box
    assert x2 > x1 and y2 > y1
    assert placed[60, 70, 0] > 150
    mixed = training_views(image, (20, 15, 145, 105), mask, copies=2, rng=np.random.default_rng(2))
    assert len(mixed) == 4
    assert np.array_equal(mixed[0][0], image)


def test_encoder_preprocess_shape_and_point_scale():
    image = np.zeros((100, 200, 3), np.uint8)
    tensor, scale = prepare_encoder_input(image, 1024)
    assert tensor.shape == (1, 3, 1024, 1024)
    assert tensor.dtype == np.float32
    assert scale_point(50, 25, scale) == (50 * scale, 25 * scale)
