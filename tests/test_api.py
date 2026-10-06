"""HTTP stand-in for the record → tap → scrubber flow."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from tapyolo.capture import encode_frame_bundle
from tapyolo.ui import TapYoloApp, make_server


def _box_at(index: int, present_until: int = 60) -> tuple[int, int, int, int] | None:
    if index >= present_until:
        return None
    x, y, bw, bh = 16 + index * 2, 40, 40, 32
    return (x, y, x + bw - 1, y + bh - 1)


def _clip() -> tuple[list[np.ndarray], float]:
    frames = []
    for index in range(90):
        image = np.full((150, 220, 3), 18, np.uint8)
        box = _box_at(index)
        if box is not None:
            x1, y1, x2, y2 = box
            image[y1 : y2 + 1, x1 : x2 + 1] = (0, 230, 0)
        frames.append(image)
    return frames, 30.0


def test_startup_loads_the_newest_trained_weights(tmp_path):
    older = tmp_path / "sessions" / "old" / "run" / "nbs" / "head" / "weights" / "best.pt"
    newer = tmp_path / "sessions" / "552ef2db5558" / "run" / "train" / "head" / "weights" / "best.pt"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    app = TapYoloApp(model="yolo11n.pt", requested_device="cpu", work_root=tmp_path)
    assert app.active_weights == str(newer)
    assert app.weight_stack == ["yolo11n.pt"]
    status = app.model_status()
    assert status["can_reset"] is True
    assert status["active_name"] == "best.pt (552ef2db5558)"


def test_ui_flow_over_http(tmp_path):
    app = TapYoloApp(model="yolo11n.pt", requested_device="cuda:0", work_root=tmp_path, epochs=12)
    httpd = make_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    base = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        assert "Record" in html
        assert ">Stop<" in html
        assert "Train Now" in html
        assert ">Inference<" in html
        assert "Reset to previous model" in html
        assert 'id="preview"' in html
        assert 'id="camera"' in html
        assert "Choose camera" in html

        with urllib.request.urlopen(base + "/api/record/frame.jpg", timeout=5) as response:
            assert response.status == 204
            assert response.read() == b""

        preview = np.zeros((24, 32, 3), np.uint8)
        preview[:, :] = (40, 180, 20)
        with app.recorder._lock:
            app.recorder._latest = preview
        with urllib.request.urlopen(base + "/api/record/frame.jpg", timeout=5) as response:
            assert response.status == 200
            jpeg = response.read()
        assert jpeg[:2] == b"\xff\xd8"
        with app.recorder._lock:
            app.recorder._latest = None

        with urllib.request.urlopen(base + "/api/model", timeout=5) as response:
            model = json.loads(response.read().decode("utf-8"))
        assert model["active_name"] == "yolo11n.pt"
        assert model["can_reset"] is False

        tuned = tmp_path / "sessions" / "abc123" / "run" / "weights" / "best.pt"
        tuned.parent.mkdir(parents=True)
        tuned.write_bytes(b"tuned")
        app.weight_stack.append(app.active_weights)
        app.active_weights = str(tuned)
        reset_request = urllib.request.Request(
            base + "/api/model/reset",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(reset_request, timeout=5) as response:
            restored = json.loads(response.read().decode("utf-8"))
        assert restored["active"] == "yolo11n.pt"
        assert restored["active_name"] == "yolo11n.pt"
        assert restored["can_reset"] is False
        failed = urllib.request.Request(
            base + "/api/model/reset",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(failed, timeout=5)
            raise AssertionError("reset should fail with no previous model")
        except urllib.error.HTTPError as exc:
            assert exc.code == 409

        with urllib.request.urlopen(base + "/api/health", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["device_requested"] == "cuda:0"
        assert health["device"] in {"cuda:0", "cpu", "mps"}
        assert health["segmenter"] == "fallback"
        assert health["onnx_weights"] is False

        frames, fps = _clip()
        payload = encode_frame_bundle(frames, fps)
        request = urllib.request.Request(
            base + "/api/sessions",
            data=payload,
            headers={"Content-Type": "application/x-tapyolo-frames"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            session = json.loads(response.read().decode("utf-8"))
        assert 20 <= session["keyframe_count"] <= 30
        assert 3 <= session["negative_count"] <= 5
        assert set(session["keyframe_indices"]).isdisjoint(session["negative_indices"])
        for path in session["negative_label_paths"]:
            assert Path(path).read_bytes() == b""

        with urllib.request.urlopen(
            f"{base}/api/sessions/{session['id']}/frames/0.jpg",
            timeout=5,
        ) as response:
            jpeg = response.read()
        assert jpeg[:2] == b"\xff\xd8"

        origin = session["keyframe_indices"][0]
        other = session["keyframe_indices"][1]
        first = _box_at(origin)
        second = _box_at(other)
        assert first is not None and second is not None
        assert first != second

        def post_box(payload: dict) -> dict:
            request = urllib.request.Request(
                f"{base}/api/sessions/{session['id']}/box",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))

        drawn = post_box({"index": 0, "x1": first[0], "y1": first[1], "x2": first[2], "y2": first[3]})
        assert drawn["bbox"] == list(first)
        assert drawn["labeled_count"] == 1
        assert drawn["boxes"][1] is None

        both = post_box({"index": 1, "x1": second[0], "y1": second[1], "x2": second[2], "y2": second[3]})
        assert both["boxes"][0] == list(first)
        assert both["boxes"][1] == list(second)
        assert both["labeled_count"] == 2

        tiny = urllib.request.Request(
            f"{base}/api/sessions/{session['id']}/box",
            data=json.dumps({"index": 2, "x1": 1, "y1": 1, "x2": 2, "y2": 2}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(tiny, timeout=5)
            raise AssertionError("tiny box should be rejected")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400

        stored = app.sessions[session["id"]]
        exported = app.export_labeled_set(stored)
        names = sorted(path.name for path in (exported / "images").iterdir())
        assert "key_000.jpg" in names
        assert "key_001.jpg" in names
        assert any(name.startswith("key_000_bg") for name in names)
        assert sum(name.startswith("key_") for name in names) > 2
        assert sum(name.startswith("neg_") for name in names) == session["negative_count"]
        label = (exported / "labels" / "key_000.txt").read_text(encoding="utf-8").split()
        assert label[0] == "0"
        assert all(0.0 <= float(value) <= 1.0 for value in label[1:])
        assert (exported / "labels" / "neg_000.txt").read_bytes() == b""
        assert (exported / "labels" / "key_000_gone.txt").read_bytes() == b""
        with urllib.request.urlopen(
            f"{base}/api/sessions/{session['id']}/masks/0.png",
            timeout=5,
        ) as response:
            png = response.read()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        httpd.shutdown()
        httpd.server_close()
