"""HTTP stand-in for the record → tap → scrubber flow."""

from __future__ import annotations

import json
import threading
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


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


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
        truth = _box_at(origin)
        assert truth is not None
        tap_x = (truth[0] + truth[2]) // 2
        tap_y = (truth[1] + truth[3]) // 2
        tap_request = urllib.request.Request(
            f"{base}/api/sessions/{session['id']}/tap",
            data=json.dumps({"index": 0, "x": tap_x, "y": tap_y}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(tap_request, timeout=30) as response:
            tapped = json.loads(response.read().decode("utf-8"))
        assert tapped["backend"] == "fallback"
        assert tapped["elapsed_ms"] >= 0
        assert len(tapped["tracks"]) == session["keyframe_count"]
        yolo = tapped["yolo"]
        assert all(0.0 <= value <= 1.0 for value in yolo)

        hits = 0
        for track, source_index in zip(tapped["tracks"], session["keyframe_indices"]):
            expected = _box_at(source_index)
            assert expected is not None
            predicted = _center(track["bbox"])
            actual = _center(expected)
            if abs(predicted[0] - actual[0]) <= 12 and abs(predicted[1] - actual[1]) <= 12:
                hits += 1
        assert hits / session["keyframe_count"] > 0.9
    finally:
        httpd.shutdown()
        httpd.server_close()
