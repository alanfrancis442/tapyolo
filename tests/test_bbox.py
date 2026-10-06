import numpy as np
import pytest

from tapyolo.bbox import mask_to_bbox, xyxy_to_yolo, yolo_to_xyxy


def test_mask_extents_follow_the_prd_minmax():
    mask = np.zeros((50, 100), np.uint8)
    mask[5:26, 10:31] = 1
    assert mask_to_bbox(mask) == (10, 5, 30, 25)
    assert mask_to_bbox(np.zeros((4, 4), np.uint8)) is None


def test_yolo_normalization_formula():
    box = xyxy_to_yolo(10, 5, 30, 25, 100, 50)
    assert box == (0.2, 0.3, 0.2, 0.4)
    restored = yolo_to_xyxy(*box, 100, 50)
    assert restored == pytest.approx((10, 5, 30, 25))


def test_normalized_values_stay_inside_unit_interval():
    box = xyxy_to_yolo(0, 0, 10, 10, 10, 10)
    assert all(0.0 <= value <= 1.0 for value in box)
