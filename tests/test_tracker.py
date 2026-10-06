import numpy as np

from tapyolo.tracker import propagate_bidirectional


def _moving_box(n=15, size=(200, 120), box=(28, 22), speed=5, origin=(16, 36)):
    width, height = size
    box_w, box_h = box
    frames = []
    truth = []
    for index in range(n):
        image = np.full((height, width, 3), 24, np.uint8)
        x = origin[0] + index * speed
        y = origin[1]
        image[y : y + box_h, x : x + box_w] = (40, 200, 50)
        frames.append(image)
        truth.append((x, y, x + box_w - 1, y + box_h - 1))
    return frames, truth


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def test_bidirectional_tracker_follows_a_moving_box():
    frames, truth = _moving_box()
    start = len(frames) // 2
    result = propagate_bidirectional(frames, start, truth[start])
    assert result.backend == "bytetrack"
    assert all(box is not None for box in result.boxes)
    hits = 0
    for box, gt in zip(result.boxes, truth):
        predicted = _center(box)
        actual = _center(gt)
        if abs(predicted[0] - actual[0]) <= 8 and abs(predicted[1] - actual[1]) <= 8:
            hits += 1
    assert hits / len(frames) > 0.9
    assert abs(_center(result.boxes[0])[0] - _center(truth[0])[0]) <= 8
    assert abs(_center(result.boxes[-1])[0] - _center(truth[-1])[0]) <= 8
