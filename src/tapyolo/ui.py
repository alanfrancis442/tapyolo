"""Local web UI for the record → tap → track → train → live flow."""

from __future__ import annotations

import json
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
  button, .file, select { background: #2a3140; color: inherit; border: 1px solid #3c465c; border-radius: 6px; padding: 8px 12px; }
  button:disabled { opacity: 0.45; }
  button.selected { background: #3dcc7a; color: #102016; border-color: #3dcc7a; }
  #modelName { color: #b7c0ce; }
  .file input { display: none; }
  main { padding: 16px; }
  #view { position: relative; width: min(960px, 100%); min-height: 360px; background: #0c0e12; }
  #preview { display: none; width: 100%; max-height: 70vh; background: #000; object-fit: contain; }
  #preview.on { display: block; min-height: 360px; }
  #stage { touch-action: none; display: none; width: 100%; height: auto; background: #0c0e12; cursor: crosshair; }
  #stage.on { display: block; }
  #overlay { display: none; position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  #overlay.on { display: block; }
  #tapHint { display: none; position: absolute; left: 12px; bottom: 12px; margin: 0; padding: 8px 12px; background: rgba(0,0,0,0.72); border-radius: 6px; pointer-events: none; }
  #tapHint.on { display: block; }
  #scrub { width: min(960px, 100%); margin-top: 8px; }
  footer { display: grid; gap: 8px; max-width: 960px; }
  label { display: flex; gap: 8px; align-items: center; }
  #label { flex: 1; background: #0c0e12; color: inherit; border: 1px solid #3c465c; border-radius: 6px; padding: 8px; }
  .bar { height: 14px; background: #2a3140; border-radius: 7px; overflow: hidden; }
  #fill { height: 100%; width: 0%; background: #3dcc7a; }
  #live { max-width: 960px; }
  #error { color: #ffb4b4; min-height: 1.2em; padding: 0 16px 16px; }
  #health, #meta, #progress, #liveStats { color: #b7c0ce; }
</style>
</head>
<body>
<header>
  <div id="rec"><span id="dot"></span> <span id="timer">00:00</span></div>
  <button id="record" type="button">Record</button>
  <button id="stop" type="button" disabled>Stop</button>
  <label>Camera <select id="camera"><option value="">Choose camera</option></select></label>
  <label class="file">Open video<input id="file" type="file" accept="video/*"></label>
  <button id="modeTrain" class="selected" type="button">Train</button>
  <button id="modeInfer" type="button">Inference</button>
  <button id="resetModel" type="button" disabled>Reset to previous model</button>
  <span id="modelName"></span>
  <span id="health"></span>
</header>
<main>
  <div id="view">
    <video id="preview" autoplay playsinline muted></video>
    <canvas id="stage" class="on" width="640" height="360"></canvas>
    <canvas id="overlay"></canvas>
    <p id="tapHint">Tap the object once. Refine with foreground/background points, or draw a correction box.</p>
  </div>
  <input id="scrub" type="range" min="0" max="0" value="0" disabled>
  <p id="meta">Record a clip, then tap the object to label its keyframes. Review the boxes before training. Remove the object for the final second if you want background samples.</p>
</main>
<footer>
  <label>Selection <select id="selectionMode">
    <option value="tap">Tap object (new selection)</option>
    <option value="add">Add foreground point</option>
    <option value="remove">Exclude background point</option>
    <option value="box">Draw correction box</option>
  </select></label>
  <button id="propagate" type="button" disabled>Propagate this correction</button>
  <label><input id="backgroundTail" type="checkbox">The object is absent during the final second</label>
  <label>Label <input id="label" placeholder="Drill Machine"></label>
  <button id="clearBox" type="button" disabled>Clear this frame</button>
  <button id="train" type="button" disabled>Train Now</button>
  <div class="bar"><div id="fill"></div></div>
  <p id="progress">Idle</p>
</footer>
<section>
  <label>Saved models <select id="savedModels"></select></label>
  <button id="useModel" type="button">Use selected model</button>
  <button id="deleteModel" type="button" disabled>Delete selected trained model</button>
  <p>Deleting a model removes its checkpoints. Captures and labels are kept.</p>
</section>
<section id="live">
  <h2>Live</h2>
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
const tapHint = document.getElementById("tapHint");
const meta = document.getElementById("meta");
const labelInput = document.getElementById("label");
const clearBtn = document.getElementById("clearBox");
const trainBtn = document.getElementById("train");
const fill = document.getElementById("fill");
const progressEl = document.getElementById("progress");
const healthEl = document.getElementById("health");
const errorEl = document.getElementById("error");
const preview = document.getElementById("preview");
const overlay = document.getElementById("overlay");
const cameraSelect = document.getElementById("camera");
const modeTrain = document.getElementById("modeTrain");
const modeInfer = document.getElementById("modeInfer");
const resetModelBtn = document.getElementById("resetModel");
const modelNameEl = document.getElementById("modelName");
const liveStats = document.getElementById("liveStats");
const snap = document.createElement("canvas");
const snapCtx = snap.getContext("2d");

let session = null;
let boxes = [];
let masks = [];
let maskTokens = [];
let frameImage = null;
let draft = null;
let drawing = false;
let drawSeq = 0;
let mode = null;
let clock = null;
let liveOn = false;
let viewMode = "train";
let serverPreview = false;
let openingCamera = null;
const selectionMode = document.getElementById("selectionMode");
const propagateBtn = document.getElementById("propagate");
const savedModels = document.getElementById("savedModels");
const deleteModelBtn = document.getElementById("deleteModel");
const useModelBtn = document.getElementById("useModel");
let pointsByFrame = {};
let availableMasks = new Set();
let annotationBusy = false;
let training = false;
let annotationPoll = null;

function startAnnotationProgress() {
  if (annotationPoll) clearInterval(annotationPoll);
  const id = session.id;
  annotationPoll = setInterval(async () => {
    try {
      const res = await fetch("/api/sessions/" + id + "/annotation-progress");
      const data = await res.json();
      if (annotationBusy && session.id === id && data.status === "running") {
        meta.textContent = data.total ? "Following the object through the video: " + data.done + "/" + data.total + " frames…" : "Preparing video tracking…";
      }
    } catch (err) { /* The primary request reports any tracking error. */ }
  }, 700);
}

function stopAnnotationProgress() {
  if (annotationPoll) clearInterval(annotationPoll);
  annotationPoll = null;
}


function updateBusyControls() {
  recordBtn.disabled = annotationBusy || training;
  fileInput.disabled = annotationBusy || training;
  clearBtn.disabled = annotationBusy || training || !boxes[Number(scrub.value)];
  selectionMode.disabled = annotationBusy || training;
  modeTrain.disabled = annotationBusy || training;
  modeInfer.disabled = annotationBusy || training;
  resetModelBtn.disabled = annotationBusy || training || !resetModelBtn.dataset.canReset;
}

let modelCatalog = [];
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
  healthEl.textContent = "Device " + data.device + "; segmenter " + data.segmenter + "; tracker " + data.tracker;
  if (!data.learned_segmentation) meta.textContent = "MobileSAM weights are unavailable; the OpenCV fallback is less reliable. Draw a box if needed. " + (data.segmenter_note || "");
}).catch((err) => showError(err.message));

openCamera("").catch((err) => {
  meta.textContent = "Choose a camera or open a video. " + err.message;
});
refreshModel();

modeTrain.addEventListener("click", () => { setMode("train"); });
modeInfer.addEventListener("click", () => { setMode("infer").catch((err) => showError(err.message)); });
resetModelBtn.addEventListener("click", () => {
  clearError();
  postJSON("/api/model/reset", {}).then((data) => {
    applyModelStatus(data);
    liveStats.textContent = "Restored " + data.active_name;
  }).catch((err) => showError(err.message));
});

savedModels.addEventListener("change", () => {
  deleteModelBtn.disabled = training || savedModels.value === "base";
});
useModelBtn.addEventListener("click", async () => {
  try { applyModelStatus(await postJSON("/api/model/select", { id: savedModels.value })); }
  catch (err) { showError(err.message); }
});
deleteModelBtn.addEventListener("click", async () => {
  const model = modelCatalog.find((item) => item.id === savedModels.value);
  if (!model || !confirm("Delete " + model.name + " and its checkpoint files? Captures and labels will be kept.")) return;
  try { applyModelStatus(await postJSON("/api/model/delete", { id: model.id })); }
  catch (err) { showError(err.message); }
});
propagateBtn.addEventListener("click", async () => {
  if (!session || annotationBusy || training) return;
  annotationBusy = true;
  scrub.disabled = true;
  updateBusyControls();
  try {
    const index = Number(scrub.value);
    startAnnotationProgress();
    const data = await postJSON("/api/sessions/" + session.id + "/propagate", { index });
    applyAnnotation(data, index);
  } catch (err) { showError(err.message); }
  finally { stopAnnotationProgress(); annotationBusy = false; scrub.disabled = false; updateBusyControls(); describeFrame(); }
});

recordBtn.addEventListener("click", async () => {
  clearError();
  try {
    await startBrowserRecord();
    mode = "browser";
  } catch (err) {
    releaseStream();
    try {
      await postJSON("/api/record/start", {});
      mode = "server";
      serverPreview = true;
      meta.textContent = "Recording. At 8 seconds the picture freezes so you can tap the object.";
      showStage();
      pollServerPreview();
      beginClock();
      recordBtn.disabled = true;
      stopBtn.disabled = false;
      cameraSelect.disabled = true;
      modeTrain.disabled = true;
      modeInfer.disabled = true;
    } catch (serverErr) {
      showError(err.message + ". " + serverErr.message);
    }
  }
});

cameraSelect.addEventListener("change", () => {
  if (recorder.running || !cameraSelect.value) return;
  clearError();
  openCamera(cameraSelect.value).catch((err) => showError(err.message));
});

cameraSelect.addEventListener("pointerdown", () => {
  if (recorder.running || cameraSelect.value) return;
  if (cameraSelect.options.length === 1 && !cameraSelect.options[0].value) {
    clearError();
    openCamera("").catch((err) => showError(err.message));
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

function showCamera() {
  preview.classList.add("on");
  stage.classList.remove("on");
  overlay.classList.remove("on");
}

function showStage() {
  preview.classList.remove("on");
  stage.classList.add("on");
  overlay.classList.remove("on");
}

function showLiveOverlay() {
  preview.classList.add("on");
  stage.classList.remove("on");
  overlay.classList.add("on");
}

function releaseStream() {
  if (recorder.stream) recorder.stream.getTracks().forEach((track) => track.stop());
  recorder.stream = null;
  preview.srcObject = null;
}

async function refreshCameras(selectedId) {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter((device) => device.kind === "videoinput");
  const previous = selectedId || cameraSelect.value;
  cameraSelect.innerHTML = "";
  if (!cams.length) {
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "No camera found";
    cameraSelect.appendChild(empty);
    return;
  }
  cams.forEach((cam, index) => {
    const opt = document.createElement("option");
    opt.value = cam.deviceId;
    opt.textContent = cam.label || ("Camera " + (index + 1));
    cameraSelect.appendChild(opt);
  });
  if (previous && [...cameraSelect.options].some((opt) => opt.value === previous)) {
    cameraSelect.value = previous;
  }
}

function openCamera(deviceId) {
  if (openingCamera) return openingCamera;
  openingCamera = openCameraNow(deviceId).finally(() => { openingCamera = null; });
  return openingCamera;
}

async function openCameraNow(deviceId) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("This browser has no camera API");
  }
  const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
  if (deviceId) video.deviceId = { exact: deviceId };
  const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video });
  releaseStream();
  preview.srcObject = stream;
  preview.muted = true;
  preview.playsInline = true;
  recorder.stream = stream;
  showCamera();
  await preview.play();
  if (preview.videoWidth && preview.videoHeight) {
    preview.style.aspectRatio = preview.videoWidth + " / " + preview.videoHeight;
  }
  const chosen = stream.getVideoTracks()[0].getSettings().deviceId || deviceId || "";
  await refreshCameras(chosen);
  meta.textContent = "Camera live. Press Record, then tap the object on a keyframe.";
  preview.scrollIntoView({ block: "nearest" });
}

async function startBrowserRecord() {
  if (!recorder.stream || !recorder.stream.active) {
    await openCamera(cameraSelect.value);
  }
  if (!preview.videoWidth) {
    await new Promise((resolve) => {
      preview.addEventListener("loadeddata", () => resolve(), { once: true });
      setTimeout(resolve, 1500);
    });
  }
  showCamera();
  tapHint.classList.remove("on");
  meta.textContent = "Recording. At 8 seconds the picture freezes so you can tap the object.";
  recorder.frames = [];
  recorder.running = true;
  recorder.started = performance.now();
  recorder.last = 0;
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  cameraSelect.disabled = true;
  modeTrain.disabled = true;
  modeInfer.disabled = true;
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
    const w = preview.videoWidth;
    const h = preview.videoHeight;
    if (!w || !h) continue;
    snap.width = w;
    snap.height = h;
    snapCtx.drawImage(preview, 0, 0, w, h);
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

function freezePreviewOntoStage() {
  const w = preview.videoWidth;
  const h = preview.videoHeight;
  if (!w || !h) return;
  stage.width = w;
  stage.height = h;
  ctx.drawImage(preview, 0, 0, w, h);
  showStage();
}

async function stopRecord() {
  stopBtn.disabled = true;
  serverPreview = false;
  if (mode !== "server") freezePreviewOntoStage();
  meta.textContent = "Clip captured. Preparing frames so you can tap the object.";
  if (mode === "server") {
    if (clock) clearInterval(clock);
    dot.classList.remove("on");
    const data = await postJSON("/api/record/stop", {});
    showSession(data);
    resetRec();
    return;
  }
  recorder.running = false;
  releaseStream();
  dot.classList.remove("on");
  if (!recorder.frames.length) {
    showError("Recording produced no frames");
    resetRec();
    return;
  }
  const bundle = buildBundle(recorder.frames, 30);
  const data = await uploadBlob(bundle, "application/x-tapyolo-frames");
  showSession(data);
  resetRec();
}

function resetRec() {
  mode = null;
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  cameraSelect.disabled = false;
  modeTrain.disabled = false;
  modeInfer.disabled = false;
  dot.classList.remove("on");
}

async function pollServerPreview() {
  while (serverPreview) {
    try {
      const res = await fetch("/api/record/frame.jpg", { cache: "no-store" });
      if (serverPreview && res.status === 200) {
        const blob = await res.blob();
        const image = await createImageBitmap(blob);
        stage.width = image.width;
        stage.height = image.height;
        ctx.drawImage(image, 0, 0);
        image.close();
      }
    } catch (err) {
      if (serverPreview) showError(err.message);
    }
    await new Promise((resolve) => setTimeout(resolve, 80));
  }
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
  showStage();
  session = data;
  boxes = new Array(data.keyframe_count).fill(null);
  pointsByFrame = {};
  availableMasks = new Set();
  masks = new Array(data.keyframe_count).fill(null);
  maskTokens = new Array(data.keyframe_count).fill("");
  frameImage = null;
  draft = null;
  scrub.disabled = data.keyframe_count === 0;
  scrub.max = Math.max(0, data.keyframe_count - 1);
  scrub.value = "0";
  tapHint.textContent = "Tap the object once. Refine with foreground/background points, or draw a correction box.";
  tapHint.classList.add("on");
  describeFrame();
  if (data.warning) meta.textContent += " " + data.warning;
  trainBtn.disabled = true;
  clearBtn.disabled = true;
  if (data.keyframe_count) drawFrame(0);
}

function describeFrame() {
  if (!session) return;
  const index = Number(scrub.value);
  const labeled = boxes.filter(Boolean).length;
  const here = boxes[index]
    ? "Review this box; add points or draw a box to correct it."
    : "No reliable track here. Tap the object or draw a correction box.";
  meta.textContent = "Frame " + (index + 1) + " of " + session.keyframe_count + ". " + here + " Use the slider to review other orientations. " + labeled + " frames labeled.";
  clearBtn.disabled = training || annotationBusy || !boxes[index];
  trainBtn.disabled = training || annotationBusy || !labelInput.value.trim() || labeled < 2;
  propagateBtn.disabled = training || annotationBusy || !boxes[index];
}

function pointerToImage(ev) {
  const rect = stage.getBoundingClientRect();
  const x = Math.round((ev.clientX - rect.left) * stage.width / Math.max(rect.width, 1));
  const y = Math.round((ev.clientY - rect.top) * stage.height / Math.max(rect.height, 1));
  return [
    Math.max(0, Math.min(stage.width - 1, x)),
    Math.max(0, Math.min(stage.height - 1, y)),
  ];
}

function strokeBox(box, color) {
  const x = box[0];
  const y = box[1];
  const w = box[2] - box[0] + 1;
  const h = box[3] - box[1] + 1;
  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(0,0,0,0.85)";
  ctx.strokeRect(x, y, w, h);
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.strokeRect(x, y, w, h);
}

function loadMask(index) {
  if (!session || !boxes[index] || !availableMasks.has(index)) return;
  const token = String(Date.now()) + "-" + index;
  maskTokens[index] = token;
  const image = new Image();
  image.onload = () => {
    if (maskTokens[index] !== token) return;
    masks[index] = image;
    if (Number(scrub.value) === index) paintStage();
  };
  image.src = "/api/sessions/" + session.id + "/masks/" + index + ".png?t=" + token;
}

function paintStage() {
  if (!frameImage) return;
  ctx.drawImage(frameImage, 0, 0);
  const overlayMask = masks[Number(scrub.value)];
  if (overlayMask) ctx.drawImage(overlayMask, 0, 0);
  for (const point of pointsByFrame[Number(scrub.value)] || []) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = point.label ? "#3dcc7a" : "#ff6565";
    ctx.fill();
  }
  const saved = boxes[Number(scrub.value)];
  if (saved) strokeBox(saved, "#3dcc7a");
  if (draft) {
    const x1 = Math.min(draft[0], draft[2]);
    const y1 = Math.min(draft[1], draft[3]);
    const x2 = Math.max(draft[0], draft[2]);
    const y2 = Math.max(draft[1], draft[3]);
    strokeBox([x1, y1, x2, y2], "#f2c14e");
  }
}

stage.addEventListener("pointerdown", (ev) => {
  if (!session || !session.keyframe_count || viewMode !== "train" || !frameImage || annotationBusy || training) return;
  if (ev.button !== undefined && ev.button !== 0) return;
  clearError();
  try { stage.setPointerCapture(ev.pointerId); } catch (err) { /* synthetic events have no capture */ }
  const [x, y] = pointerToImage(ev);
  if (selectionMode.value !== "box") {
    submitTap(x, y);
    ev.preventDefault();
    return;
  }
  masks[Number(scrub.value)] = null;
  draft = [x, y, x, y];
  drawing = true;
  paintStage();
  ev.preventDefault();
});

stage.addEventListener("pointermove", (ev) => {
  if (!drawing || !draft) return;
  const [x, y] = pointerToImage(ev);
  draft[2] = x;
  draft[3] = y;
  paintStage();
});

stage.addEventListener("pointerup", (ev) => { finishDraw(ev); });
stage.addEventListener("pointercancel", (ev) => { finishDraw(ev); });
window.addEventListener("pointermove", (ev) => {
  if (!drawing || !draft) return;
  const [x, y] = pointerToImage(ev);
  draft[2] = x;
  draft[3] = y;
  paintStage();
});
window.addEventListener("pointerup", (ev) => { if (drawing) finishDraw(ev); });

async function finishDraw(ev) {
  if (!drawing) return;
  drawing = false;
  const [x, y] = pointerToImage(ev);
  if (draft) {
    draft[2] = x;
    draft[3] = y;
  }
  const raw = draft;
  draft = null;
  paintStage();
  if (!raw) return;
  const x1 = Math.min(raw[0], raw[2]);
  const y1 = Math.min(raw[1], raw[3]);
  const x2 = Math.max(raw[0], raw[2]);
  const y2 = Math.max(raw[1], raw[3]);
  if (x2 - x1 < 8 || y2 - y1 < 8) return;
  const index = Number(scrub.value);
  annotationBusy = true;
  scrub.disabled = true;
  updateBusyControls();
  try {
    const data = await postJSON("/api/sessions/" + session.id + "/box", { index, x1, y1, x2, y2 });
    applyAnnotation(data, index);
    if (labelInput.value.trim()) labelInput.focus();
  } catch (err) {
    showError(err.message);
  } finally {
    annotationBusy = false;
    scrub.disabled = false;
    updateBusyControls();
    describeFrame();
  }
}

function applyAnnotation(data, index) {
  boxes = data.boxes;
  pointsByFrame = data.points_by_frame || pointsByFrame;
  availableMasks = new Set(data.mask_indices || []);
  masks = new Array(boxes.length).fill(null);
  maskTokens = new Array(boxes.length).fill("");
  tapHint.classList.remove("on");
  describeFrame();
  paintStage();
  loadMask(index);
}

async function submitTap(x, y) {
  const index = Number(scrub.value);
  const previous = selectionMode.value === "tap" ? [] : (pointsByFrame[index] || []);
  const points = [...previous, { x, y, label: selectionMode.value === "remove" ? 0 : 1 }];
  annotationBusy = true;
  scrub.disabled = true;
  updateBusyControls();
  describeFrame();
  meta.textContent = "Selecting and tracking the object…";
  try {
    startAnnotationProgress();
    const data = await postJSON("/api/sessions/" + session.id + "/tap", { index, points });
    applyAnnotation(data, index);
    if (selectionMode.value === "tap") selectionMode.value = "add";
    progressEl.textContent = data.backend + " + " + data.tracker + ": " + data.elapsed_ms.toFixed(0) + " ms segmentation; " + data.track_ms.toFixed(0) + " ms tracking. Review the boxes.";
  } catch (err) { showError(err.message); }
  finally { stopAnnotationProgress(); annotationBusy = false; scrub.disabled = false; updateBusyControls(); describeFrame(); }
}

scrub.addEventListener("input", () => {
  draft = null;
  drawing = false;
  describeFrame();
  drawFrame(Number(scrub.value));
  const index = Number(scrub.value);
  if (boxes[index] && !masks[index]) loadMask(index);
});

clearBtn.addEventListener("click", async () => {
  if (!session || annotationBusy || training) return;
  clearError();
  const index = Number(scrub.value);
  annotationBusy = true;
  scrub.disabled = true;
  updateBusyControls();
  try {
    const data = await postJSON("/api/sessions/" + session.id + "/box", { index, clear: true });
    applyAnnotation(data, index);
    if (!boxes.some(Boolean)) tapHint.classList.add("on");
  } catch (err) {
    showError(err.message);
  } finally {
    annotationBusy = false;
    scrub.disabled = false;
    updateBusyControls();
    describeFrame();
  }
});

labelInput.addEventListener("input", () => {
  describeFrame();
});

trainBtn.addEventListener("click", async () => {
  clearError();
  if (!session || annotationBusy || training) return;
  trainBtn.disabled = true;
  try {
    await postJSON("/api/sessions/" + session.id + "/train", {
      label: labelInput.value, tail_is_background: document.getElementById("backgroundTail").checked,
    });
    training = true;
    updateBusyControls();
    selectionMode.disabled = true;
    clearBtn.disabled = true;
    propagateBtn.disabled = true;
    useModelBtn.disabled = true;
    deleteModelBtn.disabled = true;
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
      training = false;
      updateBusyControls();
      selectionMode.disabled = false;
      useModelBtn.disabled = false;
      describeFrame();
      if (data.report) {
        if (data.report.novel_map50 != null) progressEl.textContent += " Held-out class mAP50 " + Number(data.report.novel_map50).toFixed(3) + ".";
        const ratio = data.report.anchor_retention_ratio;
        progressEl.textContent += ratio == null ? " Retention could not be measured." : " Base-class retention " + (100 * ratio).toFixed(1) + "% (target 90%).";
      }
      refreshModel();
      progressEl.textContent += " Switch to Inference to run this model.";
    }
    if (data.status === "error") {
      clearInterval(timerId);
      showError(data.message);
      training = false;
      updateBusyControls();
      selectionMode.disabled = false;
      useModelBtn.disabled = false;
      describeFrame();
      refreshModel();
    }
  }, 500);
}

function drawFrame(index) {
  const seq = ++drawSeq;
  const image = new Image();
  image.onload = () => {
    if (seq !== drawSeq) return;
    frameImage = image;
    stage.width = image.width;
    stage.height = image.height;
    paintStage();
  };
  image.src = "/api/sessions/" + session.id + "/frames/" + index + ".jpg";
}

function applyModelStatus(data) {
  modelNameEl.textContent = data.active_name;
  resetModelBtn.dataset.canReset = data.can_reset ? "yes" : "";
  resetModelBtn.disabled = !data.can_reset || data.training;
  modelCatalog = data.models || [];
  savedModels.replaceChildren();
  const base = document.createElement("option");
  base.value = "base";
  base.textContent = "Base model — " + data.base;
  savedModels.appendChild(base);
  for (const record of modelCatalog) {
    const option = document.createElement("option");
    option.value = record.id;
    option.textContent = record.name + (record.active ? " (active)" : "");
    savedModels.appendChild(option);
    if (record.active) savedModels.value = record.id;
  }
  deleteModelBtn.disabled = data.training || savedModels.value === "base";
  useModelBtn.disabled = data.training;
  resetModelBtn.textContent = data.can_reset ? "Reset to " + data.previous_name : "Reset to previous model";
}

function refreshModel() {
  return fetch("/api/model").then((r) => r.json()).then(applyModelStatus).catch((err) => showError(err.message));
}

function showTrainSurface() {
  liveOn = false;
  overlay.classList.remove("on");
  if (session && session.keyframe_count) showStage();
  else if (recorder.stream && recorder.stream.active) showCamera();
  else showStage();
}

async function setMode(mode) {
  if (recorder.running) return;
  viewMode = mode;
  modeTrain.classList.toggle("selected", mode === "train");
  modeInfer.classList.toggle("selected", mode === "infer");
  if (mode === "train") {
    showTrainSurface();
    liveStats.textContent = "";
    return;
  }
  await startLive();
}

async function startLive() {
  if (liveOn) return;
  try {
    if (!recorder.stream || !recorder.stream.active) await openCamera(cameraSelect.value);
  } catch (err) {
    liveStats.textContent = err.message;
    viewMode = "train";
    modeTrain.classList.add("selected");
    modeInfer.classList.remove("selected");
    return;
  }
  showLiveOverlay();
  liveOn = true;
  liveStats.textContent = "Starting inference…";
  inferLoop();
}

async function inferLoop() {
  while (liveOn && viewMode === "infer") {
    const blob = await new Promise((resolve) => {
      const w = preview.videoWidth;
      const h = preview.videoHeight;
      if (!w || !h) {
        resolve(null);
        return;
      }
      snap.width = w;
      snap.height = h;
      snapCtx.drawImage(preview, 0, 0, w, h);
      snap.toBlob(resolve, "image/jpeg", 0.8);
    });
    if (!blob) {
      await new Promise((resolve) => setTimeout(resolve, 40));
      continue;
    }
    const res = await fetch("/api/infer", { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob });
    const data = await res.json();
    drawLive(data);
    await new Promise((resolve) => setTimeout(resolve, 40));
  }
}

function drawLive(data) {
  const w = preview.videoWidth || overlay.width;
  const h = preview.videoHeight || overlay.height;
  overlay.width = w;
  overlay.height = h;
  const c = overlay.getContext("2d");
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
            if path == "/":
                body = INDEX_HTML.encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8")
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
