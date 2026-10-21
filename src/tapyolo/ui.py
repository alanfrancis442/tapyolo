"""Local web UI for the record → tap → track → train → live flow."""

from __future__ import annotations

import json
import mimetypes
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from tapyolo.bbox import mask_to_bbox, xyxy_to_yolo
from tapyolo.capture import SavedCapture, decode_upload, process_frames, save_capture
from tapyolo.device import resolve_device
from tapyolo.labels import clean_label, write_empty_label, write_yolo_label
from tapyolo.models import catalog, replay_sources
from tapyolo.segment import (
    SegmenterChoice,
    load_segmenter,
    scene_without_object,
    training_views,
)
from tapyolo.tracker import propagate_bidirectional
from tapyolo.train import train_detection_head
from tapyolo.video_tracker import load_video_tracker

MAX_UPLOAD_BYTES = 80 * 1024 * 1024

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def idle_progress() -> dict:
    return {
        "status": "idle",
        "epoch": 0,
        "epochs": 0,
        "fraction": 0.0,
        "message": "Idle",
        "remaining_s": None,
        "weights": None,
        "report": None,
    }


@dataclass
class Session:
    id: str
    bundle: object
    directory: Path
    saved: SavedCapture
    boxes: list | None = None
    masks: list | None = None
    label: str | None = None
    progress: dict = field(default_factory=idle_progress)
    points: dict = field(default_factory=dict)
    manual: set = field(default_factory=set)
    excluded: set = field(default_factory=set)
    scores: list = field(default_factory=list)
    tail_is_background: bool = False
    annotation_progress: dict = field(
        default_factory=lambda: {"status": "idle", "done": 0, "total": 0}
    )
    annotation_lock: object = field(default_factory=threading.RLock)


class CameraRecorder:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.fps = 30.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._cap = None
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        cv2 = _cv2()
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No camera is available to OpenCV")
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError("Camera opened but returned no frames")
        self._cap = cap
        stored = frame.copy()
        self.frames = [stored]
        with self._lock:
            self._latest = stored
        reported = cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported if 1 <= reported <= 120 else 30.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        max_frames = int(self.fps * 8)
        while self._running and len(self.frames) < max_frames:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            stored = frame.copy()
            with self._lock:
                self._latest = stored
            self.frames.append(stored)
        self._running = False

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            frame = None if self._latest is None else self._latest.copy()
        if frame is None:
            return None
        cv2 = _cv2()
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return encoded.tobytes()

    def stop(self) -> tuple[list[np.ndarray], float]:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest = None
        return list(self.frames), self.fps


class TapYoloApp:
    def __init__(
        self,
        model: str = "yolo11n.pt",
        requested_device: str = "cuda:0",
        work_root: Path | str = "runs/tapyolo",
        epochs: int = 12,
        imgsz: int = 640,
        batch: int = 0,
        anchor_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.requested_device = requested_device
        self.device = resolve_device(requested_device)
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.anchor_dir = anchor_dir
        self.training_busy = False
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        self.torch_lock = threading.Lock()
        self.segmenter_choice: SegmenterChoice = load_segmenter(device=self.device)
        self.video_tracker = load_video_tracker(device=self.device)
        self.active_weights = model
        self.weight_stack: list[str] = []
        latest = self._newest_weights()
        if latest and Path(latest).resolve() != Path(model).resolve():
            self.weight_stack.append(model)
            self.active_weights = latest
        state_file = self.work_root / "active-model.json"
        if state_file.is_file():
            try:
                saved_active = json.loads(state_file.read_text()).get("active")
            except (ValueError, OSError):
                saved_active = None
            allowed = {
                record["path"] for record in catalog(self.work_root, self.model)
            } | {self.model}
            if saved_active in allowed:
                self.active_weights = saved_active
                self.weight_stack = [] if saved_active == self.model else [self.model]
        self._yolo = None
        self._yolo_path: str | None = None
        self.recorder = CameraRecorder()

    def _newest_weights(self) -> str | None:
        if not self.work_root.is_dir():
            return None
        models = catalog(self.work_root, self.model)
        return models[0]["path"] if models else None

    def health(self) -> dict:
        choice = self.segmenter_choice
        return {
            "device_requested": self.requested_device,
            "device": self.device,
            "segmenter": choice.name,
            "onnx_weights": choice.onnx_error is None,
            "onnx_error": choice.onnx_error,
            "model": self.model,
            "learned_segmentation": choice.name != "fallback",
            "segmenter_note": choice.onnx_error,
            "tracker": self.video_tracker.name
            if self.video_tracker
            else "template fallback (install SAM 2.1 with python -m tapyolo.setup)",
        }

    def model_status(self) -> dict:
        previous = self.weight_stack[-1] if self.weight_stack else None
        return {
            "active": self.active_weights,
            "active_name": _weight_label(self.active_weights),
            "previous": previous,
            "previous_name": None if previous is None else _weight_label(previous),
            "can_reset": previous is not None,
            "models": [
                {**record, "active": record["path"] == self.active_weights}
                for record in catalog(self.work_root, self.model)
            ],
            "base": self.model,
            "training": self.training_busy,
        }

    def _persist_active(self):
        path = self.work_root / "active-model.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"active": self.active_weights}))
        temporary.replace(path)

    def select_model(self, model_id):
        with self.lock:
            if self.training_busy:
                raise ApiError(
                    409, "Wait for training to finish before changing models"
                )
            selected = (
                self.model
                if model_id == "base"
                else next(
                    (
                        r["path"]
                        for r in catalog(self.work_root, self.model)
                        if r["id"] == model_id
                    ),
                    None,
                )
            )
            if selected is None:
                raise ApiError(404, "model not found")
            with self.torch_lock:
                if selected != self.active_weights:
                    self.weight_stack.append(self.active_weights)
                    self.active_weights = selected
                    self._yolo = None
                    self._yolo_path = None
                    self._persist_active()
        return self.model_status()

    def delete_model(self, model_id):
        with self.lock:
            if self.training_busy:
                raise ApiError(
                    409, "Wait for training to finish before deleting models"
                )
            record = next(
                (r for r in catalog(self.work_root, self.model) if r["id"] == model_id),
                None,
            )
            if record is None:
                raise ApiError(404, "model not found")
            directory = Path(record["path"]).resolve().parent
            with self.torch_lock:
                deleted = {
                    str(path.resolve())
                    for path in directory.glob("*.pt")
                    if not path.is_symlink()
                }
                if str(Path(self.model).resolve()) in deleted:
                    raise ApiError(409, "The base model cannot be deleted")
                self.weight_stack = [
                    w
                    for w in self.weight_stack
                    if str(Path(w).resolve()) not in deleted
                ]
                if str(Path(self.active_weights).resolve()) in deleted:
                    self.active_weights = (
                        self.weight_stack.pop() if self.weight_stack else self.model
                    )
                self._yolo = None
                self._yolo_path = None
                for path in deleted:
                    Path(path).unlink()
                (directory / "tapyolo.json").unlink(missing_ok=True)
                self._persist_active()
        return self.model_status()

    def reset_model(self) -> dict:
        with self.lock:
            if self.training_busy:
                raise ApiError(
                    409, "Wait for training to finish before changing models"
                )
            with self.torch_lock:
                if not self.weight_stack:
                    raise ApiError(409, "No previous model to restore")
                self.active_weights = self.weight_stack.pop()
                self._persist_active()
                self._yolo = None
                self._yolo_path = None
        return self.model_status()

    def open_frames(self, frames: list[np.ndarray], fps: float) -> Session:
        bundle = process_frames(frames, fps)
        session_id = uuid.uuid4().hex[:12]
        directory = self.work_root / "sessions" / session_id
        saved = save_capture(bundle, directory)
        session = Session(
            id=session_id, bundle=bundle, directory=directory, saved=saved
        )
        session.boxes = [None] * len(bundle.keyframes)
        session.masks = [None] * len(bundle.keyframes)
        with self.lock:
            self.sessions[session_id] = session
        return session

    def session_payload(self, session: Session) -> dict:
        bundle = session.bundle
        return {
            "id": session.id,
            "fps": bundle.fps,
            "duration_s": bundle.duration_s,
            "width": bundle.width,
            "height": bundle.height,
            "keyframe_count": len(bundle.keyframes),
            "keyframe_indices": bundle.keyframe_indices,
            "negative_count": len(bundle.negatives),
            "negative_indices": bundle.negative_indices,
            "sample_method": bundle.sample_method,
            "warning": bundle.warning,
            "segmenter": self.segmenter_choice.name,
            "negative_label_paths": [
                str(path) for path in session.saved.negative_label_paths
            ],
        }

    def _require_frame(self, session: Session, index: int) -> None:
        count = len(session.bundle.keyframes)
        if session.boxes is None or len(session.boxes) != count:
            session.boxes = [None] * count
        if session.masks is None or len(session.masks) != count:
            session.masks = [None] * count
        if not 0 <= index < count:
            raise ApiError(400, "keyframe index is out of range")

    def box_state(self, session: Session, index: int | None = None) -> dict:
        stored = session.boxes or []
        boxes = [None if box is None else list(box) for box in stored]
        current = (
            None if index is None or stored[index] is None else list(stored[index])
        )
        return {
            "index": index,
            "bbox": current,
            "boxes": boxes,
            "labeled_count": sum(box is not None for box in stored),
            "points": session.points.get(index, []),
            "mask_indices": [
                i for i, mask in enumerate(session.masks or []) if mask is not None
            ],
            "scores": session.scores,
            "review": [i for i, box in enumerate(stored) if box is None],
            "points_by_frame": session.points,
        }

    def set_box(
        self, session: Session, index: int, x1: int, y1: int, x2: int, y2: int
    ) -> dict:
        if session.progress["status"] == "running":
            raise ApiError(409, "Wait for training before editing this session")
        self._require_frame(session, index)
        width, height = session.bundle.width, session.bundle.height
        left, right = sorted((int(x1), int(x2)))
        top, bottom = sorted((int(y1), int(y2)))
        left = max(0, min(width - 1, left))
        right = max(0, min(width - 1, right))
        top = max(0, min(height - 1, top))
        bottom = max(0, min(height - 1, bottom))
        if right - left < 4 or bottom - top < 4:
            raise ApiError(400, "Box is too small")
        session.boxes[index] = (left, top, right, bottom)
        # A manually corrected rectangle is authoritative. Do not cut pieces
        # away again using the color heuristic that the user is correcting.
        mask = np.zeros((height, width), np.uint8)
        mask[top : bottom + 1, left : right + 1] = 1
        session.masks[index] = mask
        session.manual.add(index)
        session.excluded.discard(index)
        session.points.pop(index, None)
        return self.box_state(session, index)

    def clear_box(self, session: Session, index: int) -> dict:
        if session.progress["status"] == "running":
            raise ApiError(409, "Wait for training before editing this session")
        self._require_frame(session, index)
        session.boxes[index] = None
        session.masks[index] = None
        session.manual.discard(index)
        session.excluded.add(index)
        session.points.pop(index, None)
        return self.box_state(session, index)

    def tap(self, session, index, points):
        if session.progress["status"] == "running":
            raise ApiError(409, "Wait for training before editing this session")
        self._require_frame(session, index)
        if not isinstance(points, list) or not 1 <= len(points) <= 32:
            raise ApiError(400, "Supply 1–32 foreground/background points")
        coords, labels = [], []
        for point in points:
            if not isinstance(point, dict):
                raise ApiError(400, "Invalid point")
            x, y, label = int(point["x"]), int(point["y"]), int(point["label"])
            if (
                not 0 <= x < session.bundle.width
                or not 0 <= y < session.bundle.height
                or label not in (0, 1)
            ):
                raise ApiError(
                    400, "Point is outside the image or has an invalid label"
                )
            coords.append([x, y])
            labels.append(label)
        if 1 not in labels:
            raise ApiError(400, "Add a foreground point first")
        started = time.perf_counter()
        try:
            with self.torch_lock:
                mask, detail = self.segmenter_choice.segmenter.segment_prompt(
                    session.bundle.keyframes[index], coords, labels
                )
        except ValueError as exc:
            raise ApiError(422, str(exc)) from exc
        bbox = mask_to_bbox(mask)
        session.boxes[index] = bbox
        session.masks[index] = mask
        session.points[index] = points
        session.manual.add(index)
        session.excluded.discard(index)
        segmentation_ms = (time.perf_counter() - started) * 1000
        tracking_started = time.perf_counter()
        state = self.propagate(session, index)
        state.update(
            {
                "elapsed_ms": segmentation_ms,
                "track_ms": (time.perf_counter() - tracking_started) * 1000,
                "backend": self.segmenter_choice.name,
                "detail": detail,
            }
        )
        return state

    def propagate(self, session, index):
        if session.progress["status"] == "running":
            raise ApiError(409, "Wait for training before editing this session")
        self._require_frame(session, index)
        box = session.boxes[index]
        if box is None:
            raise ApiError(409, "Select an object before propagating")
        session.annotation_progress = {"status": "running", "done": 0, "total": 0}
        try:
            if self.video_tracker is not None:
                dense = session.bundle.tracking_frames or session.bundle.keyframes
                mapping = (
                    session.bundle.keyframe_indices
                    if session.bundle.tracking_frames
                    else list(range(len(dense)))
                )
                mask = session.masks[index]
                if mask is None:
                    raise ApiError(
                        409, "Tap the object or draw a correction before propagating"
                    )
                corrections = {
                    mapping[offset]: session.masks[offset]
                    for offset in session.manual
                    if session.masks[offset] is not None
                }

                def progress(done, total):
                    session.annotation_progress = {
                        "status": "running",
                        "done": done,
                        "total": total,
                    }

                with self.torch_lock:
                    result = self.video_tracker.propagate(
                        dense,
                        mapping[index],
                        mask,
                        mapping,
                        corrections=corrections,
                        excluded={mapping[offset] for offset in session.excluded},
                        progress=progress,
                    )
            else:
                result = propagate_bidirectional(session.bundle.keyframes, index, box)
            session.scores = result.scores
            for offset, proposed in enumerate(result.boxes):
                if offset in session.manual or offset in session.excluded:
                    continue
                session.boxes[offset] = proposed
                session.masks[offset] = (
                    result.masks[offset] if self.video_tracker is not None else None
                )
            session.annotation_progress = {
                **session.annotation_progress,
                "status": "done",
            }
        except Exception:
            session.annotation_progress = {
                **session.annotation_progress,
                "status": "error",
            }
            raise
        state = self.box_state(session, index)
        state["tracker"] = result.backend
        state["score_type"] = (
            "mask presence, not confidence"
            if self.video_tracker
            else "template correlation"
        )
        return state

    def mask_png(self, session: Session, index: int) -> bytes:
        self._require_frame(session, index)
        mask = None if session.masks is None else session.masks[index]
        if mask is None:
            raise ApiError(404, "no segmentation for this frame")
        cv2 = _cv2()
        bgra = np.zeros((mask.shape[0], mask.shape[1], 4), np.uint8)
        bgra[mask > 0] = (122, 204, 61, 120)
        ok, encoded = cv2.imencode(".png", bgra)
        if not ok:
            raise ApiError(500, "could not encode the mask")
        return encoded.tobytes()

    def export_labeled_set(self, session: Session) -> Path:
        """Copy only the frames the user boxed, plus the empty-label tail."""
        import shutil

        if not session.boxes or not any(box is not None for box in session.boxes):
            raise ApiError(
                409, "Tap the object or draw a correction box before training"
            )
        dest = session.directory / "labeled"
        if dest.exists():
            # Older checkpoint provenance may refer to a previous export.
            dest = session.directory / ("labeled-" + uuid.uuid4().hex[:8])
        image_dir = dest / "images"
        label_dir = dest / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        width, height = session.bundle.width, session.bundle.height
        cv2 = _cv2()
        masks = session.masks or [None] * len(session.boxes)
        for path, box, mask in zip(
            session.saved.keyframe_paths, session.boxes, masks, strict=True
        ):
            if box is None:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            if mask is None or mask.shape[:2] != image.shape[:2]:
                mask = np.zeros(image.shape[:2], np.uint8)
                x1, y1, x2, y2 = box
                mask[y1 : y2 + 1, x1 : x2 + 1] = 1
            seed = zlib.adler32(path.stem.encode("utf-8")) & 0xFFFFFFFF
            views = training_views(image, box, mask, rng=np.random.default_rng(seed))
            for offset, (view, tight) in enumerate(views):
                stem = path.stem if offset == 0 else f"{path.stem}_bg{offset - 1:02d}"
                cv2.imwrite(
                    str(image_dir / f"{stem}.jpg"),
                    view,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                )
                write_yolo_label(
                    label_dir / f"{stem}.txt",
                    0,
                    xyxy_to_yolo(*tight, width, height),
                )
            gone = scene_without_object(image, mask)
            gone_stem = f"{path.stem}_gone"
            cv2.imwrite(
                str(image_dir / f"{gone_stem}.jpg"),
                gone,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            write_empty_label(label_dir / f"{gone_stem}.txt")
        for image_path, label_path in zip(
            session.saved.negative_paths if session.tail_is_background else [],
            session.saved.negative_label_paths if session.tail_is_background else [],
            strict=True,
        ):
            shutil.copyfile(image_path, image_dir / image_path.name)
            shutil.copyfile(label_path, label_dir / label_path.name)
        return dest

    def start_training(self, session: Session, label: str) -> dict:
        if not session.boxes or not any(box is not None for box in session.boxes):
            raise ApiError(
                409, "Tap the object or draw a correction box before training"
            )
        if sum(box is not None for box in session.boxes) < 2:
            raise ApiError(
                409,
                "Label at least two keyframes so training and validation are separate",
            )
        cleaned = clean_label(label)
        with self.lock:
            if self.training_busy:
                raise ApiError(409, "A training job is already running")
            self.training_busy = True
        session.label = cleaned
        session.progress = {
            "status": "running",
            "epoch": 0,
            "epochs": self.epochs,
            "fraction": 0.0,
            "message": f"Preparing {cleaned} and rehearsal images for {self.epochs} epochs.",
            "remaining_s": None,
            "weights": None,
            "report": None,
        }
        thread = threading.Thread(
            target=self._train, args=(session, cleaned), daemon=True
        )
        try:
            thread.start()
        except RuntimeError:
            with self.lock:
                self.training_busy = False
            session.progress = idle_progress()
            raise
        return session.progress

    def _train(self, session: Session, label: str) -> None:
        try:
            labeled = self.export_labeled_set(session)
            (session.directory / "classes.txt").write_text(
                label + "\n", encoding="utf-8"
            )

            def progress(epoch: int, epochs: int, remaining: float) -> None:
                session.progress = {
                    "status": "running",
                    "epoch": epoch,
                    "epochs": epochs,
                    "fraction": epoch / epochs if epochs else 0.0,
                    "message": f"Epoch {epoch}/{epochs}",
                    "remaining_s": remaining,
                    "weights": None,
                    "report": None,
                }

            from tapyolo.device import batch_for_device

            batch = batch_for_device(self.device, self.batch or None)
            with self.torch_lock:
                inherited = (
                    []
                    if self.active_weights == self.model
                    else replay_sources(self.active_weights, self.work_root)
                )
                previous_sources = [
                    (root, name)
                    for root, name in inherited
                    if root.parent.resolve() != session.directory.resolve()
                ]
                result = train_detection_head(
                    labeled,
                    label,
                    epochs=self.epochs,
                    model=self.active_weights
                    if (Path(self.active_weights).parent / "tapyolo.json").is_file()
                    else self.model,
                    device=self.device,
                    project=session.directory / "run" / uuid.uuid4().hex[:8],
                    imgsz=self.imgsz,
                    batch=batch,
                    anchor_dir=self.anchor_dir,
                    replay_sources=previous_sources,
                    progress_callback=progress,
                )
            report = None
            if result.weights:
                from tapyolo.validate import evaluate_training_run

                session.progress["message"] = (
                    "Scoring held-out frames and base-class retention"
                )
                with self.torch_lock:
                    report = evaluate_training_run(
                        self.model,
                        result.weights,
                        Path(result.dataset_root),
                        device=self.device,
                        imgsz=self.imgsz,
                    )
                metadata = {
                    "sources": [
                        {"root": str(labeled.resolve()), "name": label},
                        *[
                            {"root": str(root), "name": name}
                            for root, name in previous_sources
                        ],
                    ],
                    "report": report,
                    "epochs": result.epochs,
                }
                (Path(result.weights).parent / "tapyolo.json").write_text(
                    json.dumps(metadata, indent=2)
                )
            if result.weights and result.weights != self.active_weights:
                with self.torch_lock:
                    self.weight_stack.append(self.active_weights)
                    self.active_weights = result.weights
                    self._yolo = None
                    self._yolo_path = None
                    self._persist_active()
            session.progress = {
                "status": "done",
                "epoch": result.epochs,
                "epochs": result.epochs,
                "fraction": 1.0,
                "message": result.note,
                "remaining_s": 0,
                "weights": result.weights,
                "report": report,
                "mix_fraction": result.mix_fraction,
            }
        except Exception as exc:
            session.progress = {
                "status": "error",
                "epoch": session.progress.get("epoch", 0),
                "epochs": self.epochs,
                "fraction": session.progress.get("fraction", 0.0),
                "message": str(exc),
                "remaining_s": None,
                "weights": None,
                "report": None,
            }

        finally:
            with self.lock:
                self.training_busy = False

    def infer(self, image: np.ndarray) -> dict:
        import time

        from ultralytics import YOLO

        try:
            with self.torch_lock:
                if self._yolo is None or self._yolo_path != self.active_weights:
                    self._yolo = YOLO(self.active_weights)
                    self._yolo_path = self.active_weights
                started = time.perf_counter()
                result = self._yolo.predict(
                    image,
                    device=self.device,
                    verbose=False,
                    conf=0.15,
                    imgsz=self.imgsz,
                )[0]
                elapsed_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            return {"boxes": [], "elapsed_ms": None, "message": str(exc)}
        boxes = []
        names = result.names
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                class_id = int(box.cls)
                boxes.append(
                    {
                        "class_id": class_id,
                        "class_name": names.get(class_id, str(class_id))
                        if isinstance(names, dict)
                        else names[class_id],
                        "confidence": float(box.conf),
                        "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                    }
                )
        return {"boxes": boxes, "elapsed_ms": elapsed_ms, "message": ""}


def make_server(app: TapYoloApp, host: str, port: int) -> ThreadingHTTPServer:
    handler = _handler_class(app)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    return Server((host, port), handler)


def serve(app: TapYoloApp, host: str = "127.0.0.1", port: int = 7860) -> None:
    httpd = make_server(app, host, port)
    print(f"TapYOLO UI at http://{host}:{port}")
    print(
        f"Resolved device: {app.device} (requested {app.requested_device}). Segmenter: {app.segmenter_choice.name}."
    )
    print(f"Active model: {_weight_label(app.active_weights)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        httpd.server_close()


def _handler_class(app: TapYoloApp):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            try:
                self._get()
            except ApiError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def do_POST(self) -> None:
            try:
                self._post()
            except ApiError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def log_message(self, fmt: str, *args) -> None:
            return

        def _get(self) -> None:
            path = urlparse(self.path).path
            if path == "/" or not path.startswith("/api/"):
                relative = "index.html" if path == "/" else path.lstrip("/")
                root = WEB_DIST.resolve()
                asset = (root / relative).resolve()
                if root not in asset.parents and asset != root:
                    raise ApiError(404, "not found")
                if not asset.is_file():
                    if path == "/":
                        message = (
                            b"Build the Astro frontend first: cd web && npm install "
                            b"&& npm run build"
                        )
                        self._send(503, message, "text/plain; charset=utf-8")
                        return
                    raise ApiError(404, "not found")
                content_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                if content_type.startswith("text/") or content_type in {
                    "application/javascript",
                    "application/json",
                }:
                    content_type += "; charset=utf-8"
                self._send(200, asset.read_bytes(), content_type)
                return
            if path == "/api/health":
                self._send_json(200, app.health())
                return
            if path == "/api/model":
                self._send_json(200, app.model_status())
                return
            if path == "/api/record/frame.jpg":
                jpeg = app.recorder.latest_jpeg()
                if jpeg is None:
                    self._send(204, b"", "image/jpeg")
                    return
                self._send(200, jpeg, "image/jpeg")
                return
            if path.endswith("/annotation-progress") and path.startswith(
                "/api/sessions/"
            ):
                session = self._session_from(path.split("/")[3])
                self._send_json(200, session.annotation_progress)
                return
            if path.endswith("/progress") and path.startswith("/api/sessions/"):
                session = self._session_from(path.split("/")[3])
                self._send_json(200, session.progress)
                return
            parts = path.strip("/").split("/")
            if (
                len(parts) == 5
                and parts[:2] == ["api", "sessions"]
                and parts[3] == "masks"
            ):
                session = self._session_from(parts[2])
                name = parts[4]
                if not name.endswith(".png"):
                    raise ApiError(404, "mask not found")
                body = app.mask_png(session, int(name[:-4]))
                self._send(200, body, "image/png")
                return
            if (
                len(parts) == 5
                and parts[:2] == ["api", "sessions"]
                and parts[3] == "frames"
            ):
                session = self._session_from(parts[2])
                name = parts[4]
                if not name.endswith(".jpg"):
                    raise ApiError(404, "frame not found")
                index = int(name[:-4])
                if not 0 <= index < len(session.saved.keyframe_paths):
                    raise ApiError(404, "frame not found")
                body = session.saved.keyframe_paths[index].read_bytes()
                self._send(200, body, "image/jpeg")
                return
            if len(parts) == 3 and parts[:2] == ["api", "sessions"]:
                session = self._session_from(parts[2])
                self._send_json(200, app.session_payload(session))
                return
            raise ApiError(404, "not found")

        def _post(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/sessions":
                data = self._body()
                content_type = self.headers.get(
                    "Content-Type", "application/octet-stream"
                )
                try:
                    frames, fps = decode_upload(data, content_type)
                    session = app.open_frames(frames, fps)
                except ValueError as exc:
                    raise ApiError(400, str(exc)) from exc
                self._send_json(200, app.session_payload(session))
                return
            if path == "/api/record/start":
                try:
                    app.recorder.start()
                except Exception as exc:
                    raise ApiError(503, str(exc)) from exc
                self._send_json(200, {"status": "recording"})
                return
            if path == "/api/record/stop":
                frames, fps = app.recorder.stop()
                if not frames:
                    raise ApiError(400, "Recording produced no frames")
                session = app.open_frames(frames, fps)
                self._send_json(200, app.session_payload(session))
                return
            if path == "/api/model/select":
                self._send_json(200, app.select_model(self._json().get("id")))
                return
            if path == "/api/model/delete":
                self._send_json(200, app.delete_model(self._json().get("id")))
                return
            if path == "/api/model/reset":
                self._send_json(200, app.reset_model())
                return
            if path == "/api/infer":
                image = _decode_image(self._body())
                self._send_json(200, app.infer(image))
                return
            parts = path.strip("/").split("/")
            if (
                len(parts) == 4
                and parts[:2] == ["api", "sessions"]
                and parts[3] in {"tap", "propagate"}
            ):
                session = self._session_from(parts[2])
                payload = self._json()
                index = int(payload.get("index", 0))
                with session.annotation_lock:
                    result = (
                        app.tap(session, index, payload.get("points"))
                        if parts[3] == "tap"
                        else app.propagate(session, index)
                    )
                self._send_json(200, result)
                return
            if (
                len(parts) == 4
                and parts[:2] == ["api", "sessions"]
                and parts[3] == "box"
            ):
                session = self._session_from(parts[2])
                payload = self._json()
                index = int(payload["index"])
                if payload.get("clear"):
                    with session.annotation_lock:
                        result = app.clear_box(session, index)
                    self._send_json(200, result)
                    return
                with session.annotation_lock:
                    result = app.set_box(
                        session,
                        index,
                        int(payload["x1"]),
                        int(payload["y1"]),
                        int(payload["x2"]),
                        int(payload["y2"]),
                    )
                self._send_json(200, result)
                return
            if (
                len(parts) == 4
                and parts[:2] == ["api", "sessions"]
                and parts[3] == "train"
            ):
                session = self._session_from(parts[2])
                payload = self._json()
                try:
                    with session.annotation_lock:
                        session.tail_is_background = (
                            payload.get("tail_is_background") is True
                        )
                        progress = app.start_training(
                            session, str(payload.get("label", ""))
                        )
                except ValueError as exc:
                    raise ApiError(400, str(exc)) from exc
                self._send_json(200, progress)
                return
            raise ApiError(404, "not found")

        def _session_from(self, session_id: str) -> Session:
            if len(session_id) != 12 or any(
                ch not in "0123456789abcdef" for ch in session_id
            ):
                raise ApiError(404, "session not found")
            session = app.sessions.get(session_id)
            if session is None:
                raise ApiError(404, "session not found")
            return session

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ApiError(400, "Content-Length must be non-negative")
            if length > MAX_UPLOAD_BYTES:
                raise ApiError(413, "upload exceeds 80 MB")
            return self.rfile.read(length) if length else b""

        def _json(self) -> dict:
            raw = self._body()
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError as exc:
                raise ApiError(400, "invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ApiError(400, "invalid JSON")
            return payload

        def _send_json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                return

    return Handler


def _weight_label(path: str) -> str:
    file = Path(path)
    parts = file.parts
    if file.name == "best.pt" and "sessions" in parts:
        session_id = parts[parts.index("sessions") + 1]
        return f"best.pt ({session_id})"
    return file.name


def _decode_image(data: bytes) -> np.ndarray:
    cv2 = _cv2()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8).copy(), cv2.IMREAD_COLOR)
    if image is None:
        raise ApiError(400, "could not decode image")
    return image


def _cv2():
    import cv2

    return cv2
