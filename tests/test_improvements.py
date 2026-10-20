from pathlib import Path

import numpy as np
import pytest
import yaml

from tapyolo.models import replay_sources
from tapyolo.segment import (
    choose_prompt_mask,
    postprocess_sam_mask,
    prepare_encoder_input,
)
from tapyolo.tracker import propagate_bidirectional
from tapyolo.train import materialize_dataset, plan_repetitions
from tapyolo.ui import ApiError, TapYoloApp


def make_source(root, count=4):
    for folder in ("images", "labels"):
        (root / folder).mkdir(parents=True)
    for index in range(count):
        for suffix in ("", "_bg00", "_gone"):
            stem = f"key_{index:03d}{suffix}"
            (root / "images" / f"{stem}.jpg").write_bytes(b"image")
            (root / "labels" / f"{stem}.txt").write_text(
                "" if suffix == "_gone" else "0 0.5 0.5 0.3 0.3\n"
            )
    return root


def test_split_keeps_derivatives_out_of_validation_and_replays_classes(tmp_path):
    first, second = make_source(tmp_path / "first"), make_source(tmp_path / "second")
    dataset = tmp_path / "dataset"
    data, _ = materialize_dataset(
        [(first, "drill"), (second, "wrench")], None, dataset, ["drill", "wrench"], 1.0
    )
    config = yaml.safe_load(data.read_text())
    assert config["train"] != config["val"]
    train = list((dataset / "images/train").glob("*.jpg"))
    val = list((dataset / "images/val").glob("*.jpg"))
    assert train and val
    assert all("_bg" not in path.stem and "_gone" not in path.stem for path in val)
    assert all("key_000" not in path.stem for path in train)
    ids = {
        p.read_text().split()[0]
        for p in (dataset / "labels/train").glob("*.txt")
        if p.read_text().strip()
    }
    assert ids == {"0", "1"}


def test_one_original_cannot_validate_its_own_augmented_views(tmp_path):
    with pytest.raises(ValueError, match="two different keyframes"):
        materialize_dataset(
            [(make_source(tmp_path / "single", 1), "drill")],
            None,
            tmp_path / "dataset",
            ["drill"],
            1.0,
        )


def test_mix_repeats_anchors_when_many_augmented_novel_images():
    plan = plan_repetitions(180, 25)
    assert plan.anchor_repeats > 1
    assert abs(plan.novel_fraction - 0.7) <= 0.015


def test_sam_padding_is_zero_after_normalization_and_low_res_mask_is_unpadded():
    tensor, _ = prepare_encoder_input(np.full((50, 100, 3), 255, np.uint8), 64)
    assert np.all(tensor[:, :, 32:, :] == 0)
    logits = np.full((16, 16), -1, np.float32)
    logits[:8] = 1
    mask = postprocess_sam_mask(logits, None, (50, 100), 64)
    assert mask.mean() > 0.95


def test_mask_selection_respects_negative_points_and_rejects_whole_scene():
    good = np.zeros((30, 40), np.uint8)
    good[5:20, 10:25] = 1
    wrong = good.copy()
    wrong[1:3, 1:3] = 1
    selected = choose_prompt_mask(
        [wrong, good], [0.99, 0.8], [[15, 10], [1, 1]], [1, 0]
    )
    assert np.array_equal(selected, good)
    with pytest.raises(ValueError, match="No plausible mask"):
        choose_prompt_mask([np.ones((30, 40))], [1], [[15, 10]], [1])


def test_tracker_stops_on_disappearance():
    frames = []
    for index in range(8):
        frame = np.full((90, 140, 3), 15, np.uint8)
        if index < 4:
            frame[25:50, 20 + index * 3 : 45 + index * 3] = (0, 230, 0)
        frames.append(frame)
    result = propagate_bidirectional(frames, 0, (20, 25, 44, 49))
    assert all(box is not None for box in result.boxes[:4])
    assert all(box is None for box in result.boxes[4:])


def test_tap_propagates_and_preserves_manual_corrections_and_exclusions(tmp_path):
    app = TapYoloApp(work_root=tmp_path, requested_device="cpu")
    frames = []
    for index in range(90):
        image = np.full((90, 140, 3), 15, np.uint8)
        image[25:50, 20 + index // 3 : 45 + index // 3] = (0, 230, 0)
        frames.append(image)
    session = app.open_frames(frames, 30)
    correction = (20, 25, 44, 49)
    app.set_box(session, 1, *correction)
    app.clear_box(session, 2)
    result = app.tap(session, 0, [{"x": 30, "y": 35, "label": 1}])
    assert result["labeled_count"] > 2
    assert session.boxes[1] == correction
    assert session.boxes[2] is None
    assert result["points"][0]["label"] == 1
    assert session.masks[0][35, 30] == 1
    assert len(app.mask_png(session, 0)) > 0


def test_empty_tail_requires_confirmation_and_exports_are_immutable(tmp_path):
    app = TapYoloApp(work_root=tmp_path, requested_device="cpu")
    session = app.open_frames(
        [np.full((40, 60, 3), 15, np.uint8) for _ in range(90)], 30
    )
    app.set_box(session, 0, 10, 10, 30, 30)
    first = app.export_labeled_set(session)
    assert not list((first / "images").glob("neg_*"))
    session.tail_is_background = True
    second = app.export_labeled_set(session)
    assert first != second and first.is_dir()
    assert len(list((second / "images").glob("neg_*"))) == 4


def test_model_delete_confined_and_active_selection_persists(tmp_path):
    app = TapYoloApp(work_root=tmp_path, requested_device="cpu")
    weights = tmp_path / "sessions/abc/run/head/weights"
    weights.mkdir(parents=True)
    for name in ("best.pt", "last.pt"):
        (weights / name).write_bytes(b"checkpoint")
    capture = weights.parent.parent.parent / "capture.jpg"
    capture.write_bytes(b"keep")
    model_id = app.model_status()["models"][0]["id"]
    app.select_model(model_id)
    assert app.active_weights == str(weights / "best.pt")
    restored = TapYoloApp(work_root=tmp_path, requested_device="cpu")
    assert restored.active_weights == app.active_weights
    app.delete_model(model_id)
    assert app.active_weights == app.model
    assert not list(weights.glob("*.pt"))
    assert capture.read_bytes() == b"keep"
    assert (
        TapYoloApp(work_root=tmp_path, requested_device="cpu").active_weights
        == app.model
    )
    for invalid in ("base", "../../outside", model_id):
        with pytest.raises(ApiError) as error:
            app.delete_model(invalid)
        assert error.value.status == 404


def test_deletion_rejects_symlinked_checkpoint_and_training(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside.pt")
    outside.write_bytes(b"keep")
    weights = tmp_path / "run/weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").symlink_to(outside)
    app = TapYoloApp(work_root=tmp_path, requested_device="cpu")
    assert app.model_status()["models"] == []
    app.training_busy = True
    with pytest.raises(ApiError) as error:
        app.delete_model("anything")
    assert error.value.status == 409
    outside.unlink()


def test_replay_provenance_survives_checkpoint_deletion(tmp_path):
    import json

    source = make_source(tmp_path / "capture/labeled")
    weights = tmp_path / "run/weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    (weights / "tapyolo.json").write_text(
        json.dumps({"sources": [{"root": str(source), "name": "drill"}]})
    )
    assert replay_sources(str(checkpoint), tmp_path) == [(source, "drill")]


def test_expansion_preserves_hidden_layers_and_matches_classes_by_name():
    import torch
    from torch import nn

    from tapyolo.adaptation import expand_detector

    source = nn.Module()
    head = nn.Module()
    head.cv3 = nn.ModuleList(
        [nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.Conv2d(4, 2, 1))]
    )
    head.stride = torch.tensor([8.0])
    head.reg_max = 4
    head.nc = 2
    head.no = 18
    head.end2end = False
    source.model = nn.ModuleList([head])
    source.names = {0: "person", 1: "cup"}
    source.yaml = {"nc": 2}
    image = torch.randn(1, 3, 8, 8)
    before = head.cv3[0](image)
    expanded = expand_detector(source, ["cup", "tool", "person"])
    after = expanded.model[-1].cv3[0](image)
    assert torch.equal(before[:, 1], after[:, 0])
    assert torch.equal(before[:, 0], after[:, 2])
    assert expanded.model[-1].cv3[0][-1].in_channels == 4
    assert head.cv3[0][-1].out_channels == 2  # source was not mutated
    assert expanded.model[-1].nc == 3


def test_evaluation_compares_same_held_out_anchor_baseline(tmp_path, monkeypatch):
    import tapyolo.validate as validation

    images = tmp_path / "images/val"
    images.mkdir(parents=True)
    for name in ("anchor_one.jpg", "novel_one.jpg"):
        (images / name).write_bytes(b"image")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": ["person", "tool"]}))
    gt = [[(0, 0, 0, 10, 10), (0, 20, 20, 30, 30)], [(1, 0, 0, 10, 10)]]
    monkeypatch.setattr(validation, "_load_split", lambda *args: (gt, [None, None]))
    baseline = [[(0, 0.9, 0, 0, 10, 10), (0, 0.8, 20, 20, 30, 30)], []]
    tuned = [[(0, 0.9, 0, 0, 10, 10)], [(1, 0.9, 0, 0, 10, 10)]]
    monkeypatch.setattr(
        validation,
        "_predict",
        lambda weights, *args: baseline if weights == "base" else tuned,
    )
    report = validation.evaluate_training_run("base", "tuned", tmp_path)
    assert report["anchor_baseline_map50"] == 1
    assert report["anchor_map50"] == 0.5
    assert report["anchor_retention_ratio"] == 0.5
    assert report["meets_retention_target"] is False
    assert report["novel_map50"] == 1


def test_training_replays_previous_class_and_uses_configured_epochs(
    tmp_path, monkeypatch
):
    import json

    from tapyolo.train import TrainResult

    app = TapYoloApp(work_root=tmp_path, requested_device="cpu", epochs=13)
    old_source = make_source(tmp_path / "old/labeled")
    old_weights = tmp_path / "old/run/weights/best.pt"
    old_weights.parent.mkdir(parents=True)
    old_weights.write_bytes(b"old")
    (old_weights.parent / "tapyolo.json").write_text(
        json.dumps({"sources": [{"root": str(old_source), "name": "wrench"}]})
    )
    app.active_weights = str(old_weights)
    frames = [np.full((40, 60, 3), 15, np.uint8) for _ in range(90)]
    session = app.open_frames(frames, 30)
    app.set_box(session, 0, 10, 10, 30, 30)
    app.set_box(session, 1, 10, 10, 30, 30)
    captured = {}

    def fake_train(root, name, **kwargs):
        captured.update(kwargs)
        work = Path(kwargs["project"])
        weights = work / "train/head/weights/best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"new")
        return TrainResult(
            "data.yaml",
            str(work / "dataset"),
            str(weights.parent.parent),
            str(weights),
            "cpu",
            kwargs["epochs"],
            81,
            list(range(80)),
            0.7,
            "Trained",
        )

    monkeypatch.setattr("tapyolo.ui.train_detection_head", fake_train)
    monkeypatch.setattr(
        "tapyolo.validate.evaluate_training_run",
        lambda *args, **kwargs: {"measured": True, "anchor_retention_ratio": 0.95},
    )
    app.training_busy = True
    session.progress["status"] = "running"
    app._train(session, "drill")
    assert session.progress["status"] == "done"
    assert captured["epochs"] == 13
    assert captured["replay_sources"] == [(old_source, "wrench")]
    assert captured["model"] == str(old_weights)
    assert app.training_busy is False
    sources = replay_sources(app.active_weights, tmp_path)
    assert {name for _, name in sources} == {"wrench", "drill"}
    assert old_source.is_dir()
