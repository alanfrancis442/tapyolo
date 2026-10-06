"""Local web UI for the record → tap → track → train → live flow."""

from __future__ import annotations

import base64
import json
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from tapyolo.capture import SavedCapture, decode_upload, process_frames, save_capture
from tapyolo.device import resolve_device
from tapyolo.labels import clean_label, write_tracked_labels
from tapyolo.pipeline import tap_keyframes
from tapyolo.segment import SegmenterChoice, load_segmenter
from tapyolo.train import LOSS_MIX_NOTE, train_detection_head

MAX_UPLOAD_BYTES = 80 * 1024 * 1024

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TapYOLO</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 15px/1.4 ui-sans-serif, system-ui, sans-serif; background: #121418; color: #e8eaed; }
  header, footer, section { padding: 12px 16px; }
  header { display: flex; gap: 12px; align-items: center; background: #1b1f27; }
  #dot { width: 10px; height: 10px; border-radius: 50%; background: #666; display: inline-block; }
  #dot.on { background: #e23d3d; }
  button, .file { background: #2a3140; color: inherit; border: 1px solid #3c465c; border-radius: 6px; padding: 8px 12px; }
  button:disabled { opacity: 0.45; }
  .file input { display: none; }
  main { padding: 16px; }
  #stage { width: min(960px, 100%); height: auto; background: #0c0e12; cursor: crosshair; }
  #scrub { width: min(960px, 100%); margin-top: 8px; }
  footer { display: grid; gap: 8px; max-width: 960px; }
  label { display: flex; gap: 8px; align-items: center; }
  #label { flex: 1; background: #0c0e12; color: inherit; border: 1px solid #3c465c; border-radius: 6px; padding: 8px; }
  .bar { height: 14px; background: #2a3140; border-radius: 7px; overflow: hidden; }
  #fill { height: 100%; width: 0%; background: #3dcc7a; }
  #live { max-width: 960px; }
  .live-wrap { position: relative; width: min(640px, 100%); background: #0c0e12; }
  .live-wrap video { width: 100%; display: block; }
  #liveCanvas { position: absolute; inset: 0; width: 100%; height: 100%; }
  #error { color: #ffb4b4; min-height: 1.2em; padding: 0 16px 16px; }
  #health, #meta, #progress, #liveStats { color: #b7c0ce; }
</style>
</head>
<body>
<header>
  <div id="rec"><span id="dot"></span> <span id="timer">00:00</span></div>
  <button id="record" type="button">Record</button>
  <button id="stop" type="button" disabled>Stop</button>
  <label class="file">Open video<input id="file" type="file" accept="video/*"></label>
  <span id="health"></span>
</header>
<main>
  <canvas id="stage" width="640" height="360"></canvas>
  <input id="scrub" type="range" min="0" max="0" value="0" disabled>
  <p id="meta">Record a 3-8 second clip, or open a video. The player freezes on keyframe 0 for a tap.</p>
</main>
<footer>
  <label>Label <input id="label" placeholder="Drill Machine"></label>
  <button id="train" type="button" disabled>Train Now</button>
  <div class="bar"><div id="fill"></div></div>
  <p id="progress">Idle</p>
</footer>
<section id="live">
  <h2>Live</h2>
  <div class="live-wrap">
    <video id="liveVideo" autoplay playsinline muted></video>
    <canvas id="liveCanvas"></canvas>
  </div>
  <p id="liveStats"></p>
</section>
<p id="error"></p>
<script>
const recordBtn = document.getElementById("record");
const stopBtn = document.getElementById("stop");
const fileInput = document.getElementById("file");
const timerEl = document.getElementById("timer");
const dot = document.getElementById("dot");
const stage = document.getElementById("stage");
const ctx = stage.getContext("2d");
const scrub = document.getElementById("scrub");
const meta = document.getElementById("meta");
const labelInput = document.getElementById("label");
const trainBtn = document.getElementById("train");
const fill = document.getElementById("fill");
const progressEl = document.getElementById("progress");
const healthEl = document.getElementById("health");
const errorEl = document.getElementById("error");
const liveVideo = document.getElementById("liveVideo");
const liveCanvas = document.getElementById("liveCanvas");
const liveStats = document.getElementById("liveStats");
const snap = document.createElement("canvas");
const snapCtx = snap.getContext("2d");

let session = null;
let tracks = null;
let tapIndex = 0;
let maskB64 = null;
let drawSeq = 0;
let mode = null;
let clock = null;
let liveOn = false;
const recorder = { running: false, frames: [], last: 0, started: 0, stream: null };

function fmt(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  return String(m).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
}
function showError(message) { errorEl.textContent = message || ""; }
function clearError() { errorEl.textContent = ""; }

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

fetch("/api/health").then((r) => r.json()).then((data) => {
  healthEl.textContent = "Device " + data.device + " (requested " + data.device_requested + "); segmenter " + data.segmenter;
}).catch((err) => showError(err.message));

recordBtn.addEventListener("click", async () => {
  clearError();
  try {
    await startBrowserRecord();
    mode = "browser";
  } catch (err) {
    try {
      await postJSON("/api/record/start", {});
      mode = "server";
      beginClock();
      recordBtn.disabled = true;
      stopBtn.disabled = false;
    } catch (serverErr) {
      showError(err.message + ". " + serverErr.message);
    }
  }
});

stopBtn.addEventListener("click", () => { stopRecord().catch((err) => showError(err.message)); });

fileInput.addEventListener("change", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  clearError();
  try {
    meta.textContent = "Sampling keyframes…";
    const data = await uploadBlob(file, file.type || "application/octet-stream");
    showSession(data);
  } catch (err) {
    showError(err.message);
  }
});

async function startBrowserRecord() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("This browser has no camera API");
  }
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  liveVideo.srcObject = stream;
  await liveVideo.play();
  recorder.stream = stream;
  recorder.frames = [];
  recorder.running = true;
  recorder.started = performance.now();
  recorder.last = 0;
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  dot.classList.add("on");
  pump();
}

async function pump() {
  while (recorder.running) {
    const now = performance.now();
    const wait = Math.max(0, (1000 / 30) - (now - recorder.last));
    await new Promise((resolve) => setTimeout(resolve, wait));
    if (!recorder.running) break;
    recorder.last = performance.now();
    const w = liveVideo.videoWidth || 640;
    const h = liveVideo.videoHeight || 480;
    snap.width = w;
    snap.height = h;
    snapCtx.drawImage(liveVideo, 0, 0, w, h);
    const blob = await new Promise((resolve) => snap.toBlob(resolve, "image/jpeg", 0.92));
    if (blob && recorder.running) recorder.frames.push(await blob.arrayBuffer());
    timerEl.textContent = fmt((performance.now() - recorder.started) / 1000);
    if ((performance.now() - recorder.started) / 1000 >= 8) {
      await stopRecord();
      break;
    }
  }
}

function beginClock() {
  const started = performance.now();
  dot.classList.add("on");
  clock = setInterval(() => { timerEl.textContent = fmt((performance.now() - started) / 1000); }, 200);
}

async function stopRecord() {
  stopBtn.disabled = true;
  if (mode === "server") {
    if (clock) clearInterval(clock);
    dot.classList.remove("on");
    const data = await postJSON("/api/record/stop", {});
    showSession(data);
    resetRec();
    return;
  }
  recorder.running = false;
  if (recorder.stream) recorder.stream.getTracks().forEach((track) => track.stop());
  dot.classList.remove("on");
  if (!recorder.frames.length) {
    showError("Recording produced no frames");
    resetRec();
    return;
  }
  meta.textContent = "Sampling keyframes…";
  const bundle = buildBundle(recorder.frames, 30);
  const data = await uploadBlob(bundle, "application/x-tapyolo-frames");
  showSession(data);
  resetRec();
}

function resetRec() {
  mode = null;
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  dot.classList.remove("on");
}

function buildBundle(frames, fps) {
  const chunks = [];
  const header = new Uint8Array(12);
  const view = new DataView(header.buffer);
  header[0] = 84; header[1] = 89; header[2] = 70; header[3] = 82;
  view.setUint32(4, frames.length, false);
  view.setUint32(8, Math.round(fps * 1000), false);
  chunks.push(header);
  for (const frame of frames) {
    const len = new Uint8Array(4);
    new DataView(len.buffer).setUint32(0, frame.byteLength, false);
    chunks.push(len);
    chunks.push(new Uint8Array(frame));
  }
  return new Blob(chunks, { type: "application/x-tapyolo-frames" });
}

async function uploadBlob(blob, contentType) {
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": contentType },
    body: blob,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "upload failed");
  return data;
}

function showSession(data) {
  session = data;
  tracks = null;
  maskB64 = null;
  scrub.disabled = data.keyframe_count === 0;
  scrub.max = Math.max(0, data.keyframe_count - 1);
  scrub.value = "0";
  let text = data.keyframe_count + " keyframes via " + data.sample_method + "; " + data.negative_count + " empty-label negatives; " + Number(data.duration_s).toFixed(2) + "s";
  if (data.warning) text += ". " + data.warning;
  meta.textContent = text;
  trainBtn.disabled = true;
  if (data.keyframe_count) drawFrame(0, null, null);
}

stage.addEventListener("click", async (ev) => {
  if (!session || !session.keyframe_count) return;
  clearError();
  const rect = stage.getBoundingClientRect();
  const x = Math.round((ev.clientX - rect.left) * stage.width / rect.width);
  const y = Math.round((ev.clientY - rect.top) * stage.height / rect.height);
  const index = Number(scrub.value);
  try {
    const data = await postJSON("/api/sessions/" + session.id + "/tap", { index, x, y });
    tracks = data.tracks;
    tapIndex = index;
    maskB64 = data.mask_png_base64;
    meta.textContent = "Segmenter " + data.backend + " (" + data.detail + ") in " + Number(data.elapsed_ms).toFixed(1) + " ms; tracker " + data.tracker + " in " + Number(data.track_ms).toFixed(1) + " ms";
    drawFrame(index, data.tracks[index].bbox, maskB64);
    trainBtn.disabled = !labelInput.value.trim();
  } catch (err) {
    showError(err.message);
  }
});

scrub.addEventListener("input", () => {
  const index = Number(scrub.value);
  const box = tracks ? tracks[index].bbox : null;
  drawFrame(index, box, index === tapIndex ? maskB64 : null);
});

labelInput.addEventListener("input", () => {
  trainBtn.disabled = !labelInput.value.trim() || !tracks;
});

trainBtn.addEventListener("click", async () => {
  clearError();
  trainBtn.disabled = true;
  try {
    await postJSON("/api/sessions/" + session.id + "/train", { label: labelInput.value });
    poll();
  } catch (err) {
    showError(err.message);
    trainBtn.disabled = false;
  }
});

function poll() {
  const timerId = setInterval(async () => {
    const res = await fetch("/api/sessions/" + session.id + "/progress");
    const data = await res.json();
    fill.style.width = (Number(data.fraction) * 100) + "%";
    let text = data.message || data.status;
    if (data.remaining_s != null) text += " (" + Math.ceil(data.remaining_s) + "s remaining)";
    progressEl.textContent = text;
    if (data.status === "done") {
      clearInterval(timerId);
      if (data.report && data.report.measured && data.report.anchor_map50 != null) {
        progressEl.textContent += " Anchor mAP50 " + Number(data.report.anchor_map50).toFixed(3) + ".";
      }
      startLive().catch((err) => showError(err.message));
    }
    if (data.status === "error") {
      clearInterval(timerId);
      showError(data.message);
      trainBtn.disabled = false;
    }
  }, 500);
}

function drawFrame(index, box, mask) {
  const seq = ++drawSeq;
  const image = new Image();
  image.onload = () => {
    if (seq !== drawSeq) return;
    stage.width = image.width;
    stage.height = image.height;
    ctx.drawImage(image, 0, 0);
    const finish = () => {
      if (!box) return;
      ctx.strokeStyle = "#3dcc7a";
      ctx.lineWidth = 2;
      ctx.strokeRect(box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1);
    };
    if (mask) {
      const overlay = new Image();
      overlay.onload = () => {
        if (seq !== drawSeq) return;
        ctx.drawImage(overlay, 0, 0);
        finish();
      };
      overlay.src = "data:image/png;base64," + mask;
    } else finish();
  };
  image.src = "/api/sessions/" + session.id + "/frames/" + index + ".jpg";
}

async function startLive() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    liveStats.textContent = "Live camera needs a browser camera. Tuned weights are ready for /api/infer.";
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  liveVideo.srcObject = stream;
  await liveVideo.play();
  liveOn = true;
  while (liveOn) {
    const blob = await new Promise((resolve) => {
      const w = liveVideo.videoWidth || 640;
      const h = liveVideo.videoHeight || 480;
      snap.width = w;
      snap.height = h;
      snapCtx.drawImage(liveVideo, 0, 0, w, h);
      snap.toBlob(resolve, "image/jpeg", 0.8);
    });
    const res = await fetch("/api/infer", { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob });
    const data = await res.json();
    drawLive(data);
    await new Promise((resolve) => setTimeout(resolve, 40));
  }
}

function drawLive(data) {
  const w = liveVideo.videoWidth || liveCanvas.width;
  const h = liveVideo.videoHeight || liveCanvas.height;
  liveCanvas.width = w;
  liveCanvas.height = h;
  const c = liveCanvas.getContext("2d");
  c.clearRect(0, 0, w, h);
  c.strokeStyle = "#3dcc7a";
  c.fillStyle = "#3dcc7a";
  c.lineWidth = 2;
  c.font = "16px sans-serif";
  for (const box of data.boxes || []) {
    const xy = box.xyxy;
    c.strokeRect(xy[0], xy[1], xy[2] - xy[0], xy[3] - xy[1]);
    c.fillText(box.class_name + " " + Number(box.confidence).toFixed(2), xy[0], Math.max(16, xy[1] - 4));
  }
  if (data.message) liveStats.textContent = data.message;
  else if (data.elapsed_ms != null) liveStats.textContent = "Inference " + Number(data.elapsed_ms).toFixed(1) + " ms";
}
</script>
</body>
</html>
"""


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
    tracks: list | None = None
    tap_index: int | None = None
    label: str | None = None
    progress: dict = field(default_factory=idle_progress)


class CameraRecorder:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.fps = 30.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._cap = None

    def start(self) -> None:
        cv2 = _cv2()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No camera is available to OpenCV")
        self._cap = cap
        self.frames = []
        reported = cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported if 1 <= reported <= 120 else 30.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        max_frames = int(self.fps * 8)
        while self._running and len(self.frames) < max_frames:
            ok, frame = self._cap.read()
            if not ok:
                break
            self.frames.append(frame)
        self._running = False

    def stop(self) -> tuple[list[np.ndarray], float]:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
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
    ) -> None:
        self.model = model
        self.requested_device = requested_device
        self.device = resolve_device(requested_device)
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        self.torch_lock = threading.Lock()
        self.segmenter_choice: SegmenterChoice = load_segmenter()
        self.active_weights: str | None = None
        self._yolo = None
        self._yolo_path: str | None = None
        self.recorder = CameraRecorder()

    def health(self) -> dict:
        choice = self.segmenter_choice
        return {
            "device_requested": self.requested_device,
            "device": self.device,
            "segmenter": choice.name,
            "onnx_weights": choice.onnx_error is None,
            "onnx_error": choice.onnx_error,
            "model": self.model,
        }

    def open_frames(self, frames: list[np.ndarray], fps: float) -> Session:
        bundle = process_frames(frames, fps)
        session_id = uuid.uuid4().hex[:12]
        directory = self.work_root / "sessions" / session_id
        saved = save_capture(bundle, directory)
        session = Session(id=session_id, bundle=bundle, directory=directory, saved=saved)
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
            "negative_label_paths": [str(path) for path in session.saved.negative_label_paths],
        }

    def tap(self, session: Session, index: int, x: int, y: int) -> dict:
        result = tap_keyframes(session.bundle.keyframes, index, x, y, self.segmenter_choice.segmenter)
        session.tracks = result["tracks"]
        session.tap_index = index
        mask = result.pop("mask")
        result["mask_png_base64"] = _mask_png(mask)
        result["frame_index"] = index
        return result

    def start_training(self, session: Session, label: str) -> dict:
        if not session.tracks:
            raise ApiError(409, "Tap a target before training")
        cleaned = clean_label(label)
        session.label = cleaned
        session.progress = {
            "status": "running",
            "epoch": 0,
            "epochs": self.epochs,
            "fraction": 0.0,
            "message": LOSS_MIX_NOTE,
            "remaining_s": None,
            "weights": None,
            "report": None,
        }
        thread = threading.Thread(target=self._train, args=(session, cleaned), daemon=True)
        thread.start()
        return session.progress

    def _train(self, session: Session, label: str) -> None:
        try:
            width, height = session.bundle.width, session.bundle.height
            boxes = [None if track["bbox"] is None else tuple(track["bbox"]) for track in session.tracks]
            stems = [path.stem for path in session.saved.keyframe_paths]
            write_tracked_labels(session.directory / "labels", stems, boxes, 0, width, height)
            (session.directory / "classes.txt").write_text(label + "\n", encoding="utf-8")

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
                result = train_detection_head(
                    session.directory,
                    label,
                    epochs=self.epochs,
                    model=self.model,
                    device=self.device,
                    project=session.directory / "run",
                    imgsz=self.imgsz,
                    batch=batch,
                    progress_callback=progress,
                )
            report = None
            if result.weights:
                self.active_weights = result.weights
                self._yolo = None
                try:
                    from tapyolo.validate import evaluate_retention

                    anchor_dir = session.directory / "run" / "anchors"
                    report = evaluate_retention(
                        result.weights,
                        anchor_dir,
                        session.directory,
                        result.novel_class_id,
                        device=self.device,
                        imgsz=self.imgsz,
                    )
                except Exception as exc:
                    report = {"measured": False, "meets_90_map": False, "error": str(exc)}
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

    def infer(self, image: np.ndarray) -> dict:
        if not self.active_weights:
            return {"boxes": [], "elapsed_ms": None, "message": "Tuned weights are not available yet"}
        import time

        from ultralytics import YOLO

        with self.torch_lock:
            if self._yolo is None or self._yolo_path != self.active_weights:
                self._yolo = YOLO(self.active_weights)
                self._yolo_path = self.active_weights
            started = time.perf_counter()
            result = self._yolo.predict(
                image, device=self.device, verbose=False, conf=0.25, imgsz=self.imgsz
            )[0]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        boxes = []
        names = result.names
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                class_id = int(box.cls)
                boxes.append(
                    {
                        "class_id": class_id,
                        "class_name": names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id],
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
    print(f"Resolved device: {app.device} (requested {app.requested_device}). Segmenter: {app.segmenter_choice.name}.")
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
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def do_POST(self) -> None:
            try:
                self._post()
            except ApiError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def log_message(self, fmt: str, *args) -> None:
            return

        def _get(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                body = INDEX_HTML.encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8")
                return
            if path == "/api/health":
                self._send_json(200, app.health())
                return
            if path.endswith("/progress") and path.startswith("/api/sessions/"):
                session = self._session_from(path.split("/")[3])
                self._send_json(200, session.progress)
                return
            parts = path.strip("/").split("/")
            if len(parts) == 5 and parts[:2] == ["api", "sessions"] and parts[3] == "frames":
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
                content_type = self.headers.get("Content-Type", "application/octet-stream")
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
            if path == "/api/infer":
                image = _decode_image(self._body())
                self._send_json(200, app.infer(image))
                return
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "tap":
                session = self._session_from(parts[2])
                payload = self._json()
                try:
                    result = app.tap(session, int(payload["index"]), int(payload["x"]), int(payload["y"]))
                except ValueError as exc:
                    raise ApiError(400, str(exc)) from exc
                self._send_json(200, result)
                return
            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "train":
                session = self._session_from(parts[2])
                payload = self._json()
                try:
                    progress = app.start_training(session, str(payload.get("label", "")))
                except ValueError as exc:
                    raise ApiError(400, str(exc)) from exc
                self._send_json(200, progress)
                return
            raise ApiError(404, "not found")

        def _session_from(self, session_id: str) -> Session:
            if len(session_id) != 12 or any(ch not in "0123456789abcdef" for ch in session_id):
                raise ApiError(404, "session not found")
            session = app.sessions.get(session_id)
            if session is None:
                raise ApiError(404, "session not found")
            return session

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0"))
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


def _mask_png(mask: np.ndarray) -> str:
    cv2 = _cv2()
    height, width = mask.shape[:2]
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[mask > 0] = (80, 220, 40, 120)
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _decode_image(data: bytes) -> np.ndarray:
    cv2 = _cv2()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8).copy(), cv2.IMREAD_COLOR)
    if image is None:
        raise ApiError(400, "could not decode image")
    return image


def _cv2():
    import cv2

    return cv2
