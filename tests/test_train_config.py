from pathlib import Path

from tapyolo.anchors import ANCHOR_COUNT, ANCHOR_NAMES, generate_anchor_set
from tapyolo.train import (
    LABELED_EPOCHS,
    LABELED_LR0,
    LOSS_MIX_NOTE,
    TrainHyperParams,
    build_train_overrides,
    plan_repetitions,
    train_detection_head,
    train_labeled_class,
)


def test_default_freeze_and_augmentations():
    params = TrainHyperParams()
    assert params.freeze == 10
    assert params.lr0 == 0.001
    assert 10 <= params.epochs <= 15
    overrides = build_train_overrides(
        params,
        data="data.yaml",
        device="cpu",
        project="runs",
        imgsz=640,
        batch=8,
    )
    assert overrides["freeze"] == 10
    assert overrides["epochs"] == 12
    assert overrides["lr0"] == 0.001
    assert overrides["mosaic"] == 0.5
    assert overrides["hsv_h"] == 0.015
    assert overrides["hsv_s"] == 0.7
    assert overrides["hsv_v"] == 0.4
    assert overrides["scale"] == 0.5
    assert overrides["degrees"] == 10
    assert overrides["close_mosaic"] == 0
    assert overrides["optimizer"] == "AdamW"


def test_mix_fraction_is_near_mu():
    plan = plan_repetitions(20, 25, 0.70)
    assert abs(plan.novel_fraction - 0.70) < 0.05
    tiny = plan_repetitions(1, 25, 0.70)
    assert abs(tiny.novel_fraction - 0.70) < 0.05


def test_anchor_set_has_25_multiclass_images(tmp_path):
    root = generate_anchor_set(tmp_path / "anchors")
    images = list((root / "images").glob("*.jpg"))
    assert len(images) == ANCHOR_COUNT
    multi = 0
    for image in images:
        lines = (
            (root / "labels" / f"{image.stem}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert lines
        if len(lines) >= 2:
            multi += 1
    assert multi > 0
    assert (root / "classes.txt").read_text(encoding="utf-8").split() == ANCHOR_NAMES


def test_train_function_accepts_a_tiny_epoch_budget(tmp_path):
    novel = tmp_path / "novel"
    (novel / "images").mkdir(parents=True)
    (novel / "labels").mkdir()
    (novel / "images" / "key_000.jpg").write_bytes(b"not-a-real-jpeg-but-copied")
    (novel / "labels" / "key_000.txt").write_text(
        "0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8"
    )
    (novel / "images" / "key_001.jpg").write_bytes(b"jpeg")
    (novel / "labels" / "key_001.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    captured: dict = {}

    class FakeYOLO:
        def __init__(self, model):
            captured["model"] = model
            self.trainer = None

        def add_callback(self, event, fn):
            captured.setdefault("callbacks", []).append(event)

        def train(self, **kwargs):
            captured["kwargs"] = kwargs

    result = train_detection_head(
        novel,
        "wrench",
        epochs=1,
        model="yolo11n.pt",
        device="cpu",
        project=tmp_path / "run",
        imgsz=64,
        batch=2,
        yolo_factory=FakeYOLO,
        anchor_dir=generate_anchor_set(tmp_path / "anchors"),
    )
    assert captured["model"] == "yolo11n.pt"
    assert "on_train_epoch_end" in captured["callbacks"]
    kwargs = captured["kwargs"]
    assert kwargs["freeze"] == 10
    assert kwargs["epochs"] == 1
    assert kwargs["lr0"] == 0.001
    assert kwargs["mosaic"] == 0.5
    assert kwargs["hsv_h"] == 0.015
    assert kwargs["hsv_s"] == 0.7
    assert kwargs["hsv_v"] == 0.4
    assert kwargs["scale"] == 0.5
    assert kwargs["degrees"] == 10
    data = Path(kwargs["data"]).read_text(encoding="utf-8")
    assert "wrench" in data
    assert "person" in data
    images = list((tmp_path / "run" / "dataset" / "images" / "train").glob("*"))
    novel_images = [path for path in images if path.name.startswith("novel_")]
    anchor_images = [path for path in images if path.name.startswith("anchor_")]
    fraction = len(novel_images) / (len(novel_images) + len(anchor_images))
    assert abs(fraction - 0.70) < 0.05
    assert result.mix_fraction == fraction
    assert "0.70" in LOSS_MIX_NOTE
    sample = next(
        (tmp_path / "run" / "dataset" / "labels" / "train").glob("novel_*.txt")
    )
    assert sample.read_text(encoding="utf-8").startswith("5 ")


def test_labeled_class_keeps_the_learning_rate(tmp_path):
    novel = tmp_path / "labeled"
    (novel / "images").mkdir(parents=True)
    (novel / "labels").mkdir()
    (novel / "images" / "key_000.jpg").write_bytes(b"jpeg")
    (novel / "labels" / "key_000.txt").write_text(
        "0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8"
    )
    (novel / "images" / "key_001.jpg").write_bytes(b"jpeg")
    (novel / "labels" / "key_001.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    captured: dict = {}

    class FakeYOLO:
        def __init__(self, model):
            captured["model"] = model
            self.trainer = None

        def add_callback(self, event, fn):
            captured.setdefault("callbacks", []).append(event)

        def train(self, **kwargs):
            captured["kwargs"] = kwargs

    result = train_labeled_class(
        novel,
        "ameen",
        model="yolo11n.pt",
        device="cpu",
        project=tmp_path / "run",
        imgsz=64,
        batch=4,
        yolo_factory=FakeYOLO,
    )
    kwargs = captured["kwargs"]
    assert captured["model"] == "yolo11n.pt"
    assert Path(kwargs["project"]).is_absolute()
    assert kwargs["project"].endswith("/train")
    assert kwargs["epochs"] == LABELED_EPOCHS
    assert kwargs["lr0"] == LABELED_LR0
    assert kwargs["nbs"] == 4
    assert kwargs["batch"] == 4
    assert kwargs["freeze"] == 10
    assert kwargs["mosaic"] == 0.0
    assert kwargs["mixup"] == 0.0
    data = Path(kwargs["data"]).read_text(encoding="utf-8")
    assert "ameen" in data
    assert "person" not in data
    assert "images/val" in data
    assert "images/train" in data
    assert result.novel_class_id == 0
    assert result.mix_fraction == 1.0
    assert result.weights is None
