from types import SimpleNamespace

import cv2
import numpy as np
import torch

from tapyolo.capture import process_frames
from tapyolo.ui import TapYoloApp
from tapyolo.video_tracker import Sam2VideoTracker, VideoTrackResult, checked_mask


def test_capture_retains_intervening_frames_at_bounded_resolution():
    frames = [np.full((720, 1280, 3), index, np.uint8) for index in range(60)]
    bundle = process_frames(frames, 30)
    assert (
        len(bundle.tracking_frames) == 30
    )  # one second of content, one second reserved tail
    assert bundle.tracking_frames[0].shape == (360, 640, 3)
    for sampled, original in enumerate(bundle.keyframe_indices):
        assert int(bundle.tracking_frames[original][0, 0, 0]) == int(
            bundle.keyframes[sampled][0, 0, 0]
        )


def test_video_tracker_maps_both_directions_and_keeps_masks_and_corrections(tmp_path):
    class FakePredictor:
        def __init__(self):
            self.inference_state = {}
            self.corrections = {}
            self.visited = []

        def reset_image(self):
            pass

        def __call__(self, source, stream, masks):
            cap = cv2.VideoCapture(source)
            count = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # The frame's brightness identifies its original source index.
                    index = round(float(frame.mean()) / 30)
                    self.visited.append(index)
                    count += 1
                    mask = np.zeros(frame.shape[:2], np.uint8)
                    mask[12:32, 10 + index : 20 + index] = 1
                    if count in self.corrections:
                        mask = self.corrections[count]
                    yield SimpleNamespace(
                        masks=SimpleNamespace(data=torch.from_numpy(mask[None]))
                    )
            finally:
                cap.release()

    predictor = FakePredictor()
    tracker = Sam2VideoTracker(
        tmp_path / "fake.pt", predictor_factory=lambda *args: predictor
    )
    frames = [np.full((48, 64, 3), index * 30, np.uint8) for index in range(6)]
    seed = np.zeros((48, 64), np.uint8)
    seed[12:32, 12:22] = 1
    corrected = np.zeros_like(seed)
    corrected[10:30, 20:30] = 1
    progress = []
    result = tracker.propagate(
        frames,
        2,
        seed,
        [0, 2, 4, 5],
        corrections={4: corrected},
        excluded={5},
        progress=lambda done, total: progress.append((done, total)),
    )
    assert predictor.visited == [2, 3, 4, 5, 2, 1, 0]
    assert result.backend == "sam2-video"
    assert result.boxes[0] == (10, 12, 19, 31)
    assert np.array_equal(result.masks[1], seed)
    assert np.array_equal(result.masks[2], corrected)
    assert result.boxes[3] is None
    assert progress[-1] == (7, 7)
    assert predictor.inference_state == {} and predictor.corrections == {}


def test_mask_gate_rejects_scene_and_discontinuous_jump():
    seed = np.zeros((200, 300), np.uint8)
    seed[10:30, 10:30] = 1
    jumped = np.zeros_like(seed)
    jumped[160:180, 260:280] = 1
    assert checked_mask(np.ones_like(seed), seed.shape, 400, seed) is None
    assert checked_mask(jumped, seed.shape, 400, seed) is None
    assert checked_mask(np.zeros_like(seed), seed.shape, 400, seed) is None
    assert np.array_equal(checked_mask(seed, seed.shape, 400, seed), seed)


def test_ui_uses_dense_video_and_exports_propagated_masks(tmp_path):
    app = TapYoloApp(work_root=tmp_path, requested_device="cpu")
    frames = [np.full((48, 64, 3), 15, np.uint8) for _ in range(90)]
    session = app.open_frames(frames, 30)
    app.set_box(session, 0, 10, 12, 20, 32)
    app.clear_box(session, 2)
    captured = {}

    class FakeTracker:
        name = "sam2-video"

        def propagate(self, dense, origin, mask, mapping, **kwargs):
            captured.update(
                count=len(dense),
                origin=origin,
                mapping=mapping,
                excluded=kwargs["excluded"],
            )
            kwargs["progress"](len(dense), len(dense))
            mask_list = [mask.copy() for _ in mapping]
            boxes = [(10, 12, 20, 32) for _ in mapping]
            return VideoTrackResult(boxes, mask_list, [1.0] * len(mapping))

    app.video_tracker = FakeTracker()
    result = app.propagate(session, 0)
    assert captured["count"] == 60  # not just the 20 sampled keyframes
    assert captured["mapping"] == session.bundle.keyframe_indices
    assert session.bundle.keyframe_indices[2] in captured["excluded"]
    assert session.boxes[2] is None
    assert session.masks[1] is not None
    assert len(app.mask_png(session, 1)) > 0
    assert result["tracker"] == "sam2-video"
    assert session.annotation_progress["status"] == "done"
