from tapyolo.labels import (
    format_yolo_line,
    write_empty_label,
    write_tracked_labels,
    write_yolo_label,
)


def test_yolo_label_line(tmp_path):
    path = tmp_path / "a.txt"
    write_yolo_label(path, 3, (0.5, 0.25, 0.2, 0.1))
    text = path.read_text(encoding="utf-8")
    assert text == format_yolo_line(3, (0.5, 0.25, 0.2, 0.1))
    class_id, *values = text.split()
    assert class_id == "3"
    assert [float(value) for value in values] == [0.5, 0.25, 0.2, 0.1]


def test_empty_negative_label(tmp_path):
    path = tmp_path / "labels" / "neg_000.txt"
    write_empty_label(path)
    assert path.read_bytes() == b""


def test_tracked_labels_skip_missing_boxes(tmp_path):
    paths = write_tracked_labels(
        tmp_path,
        ["key_000", "key_001"],
        [(10, 5, 30, 25), None],
        0,
        100,
        50,
    )
    assert paths[0].read_text(encoding="utf-8").startswith("0 0.200000 0.300000")
    assert paths[1].read_bytes() == b""
