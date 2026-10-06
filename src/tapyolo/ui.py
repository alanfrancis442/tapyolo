"""Local web UI for the record → tap → track → train → live flow."""

from __future__ import annotations

import json
import threading
import uuid
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from tapyolo.bbox import xyxy_to_yolo
from tapyolo.capture import SavedCapture, decode_upload, process_frames, save_capture
from tapyolo.device import resolve_device
from tapyolo.labels import clean_label, write_empty_label, write_yolo_label
from tapyolo.segment import SegmenterChoice, load_segmenter, object_mask, scene_without_object, training_views
from tapyolo.train import LABELED_EPOCHS, train_labeled_class

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
  #stage { display: none; width: 100%; height: auto; background: #0c0e12; cursor: crosshair; }
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
    <p id="tapHint">Drag a box around the object. Green is the part training keeps.</p>
  </div>
  <input id="scrub" type="range" min="0" max="0" value="0" disabled>
  <p id="meta">Record a clip. Drag a box around the object on each orientation. Green marks the segmented object, and training leaves the background out.</p>
</main>
<footer>
  <label>Label <input id="label" placeholder="Drill Machine"></label>
  <button id="clearBox" type="button" disabled>Clear this frame</button>
  <button id="train" type="button" disabled>Train Now</button>
  <div class="bar"><div id="fill"></div></div>
  <p id="progress">Idle</p>
</footer>
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
      meta.textContent = "Recording. At 8 seconds the picture freezes so you can draw a box.";
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
  meta.textContent = "Camera live. Press Record, then drag a box around the object on each orientation.";
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
  meta.textContent = "Recording. At 8 seconds the picture freezes so you can draw a box.";
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
  meta.textContent = "Clip captured. Preparing frames so you can draw boxes.";
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
  masks = new Array(data.keyframe_count).fill(null);
  maskTokens = new Array(data.keyframe_count).fill("");
  frameImage = null;
  draft = null;
  scrub.disabled = data.keyframe_count === 0;
  scrub.max = Math.max(0, data.keyframe_count - 1);
  scrub.value = "0";
  tapHint.textContent = "Drag a box around the object. Green is the part training keeps.";
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
    ? "Green marks the object. Training leaves the rest of this frame out."
    : "Drag a rectangle around the object on this frame.";
  meta.textContent = "Frame " + (index + 1) + " of " + session.keyframe_count + ". " + here + " Use the slider to box other orientations. " + labeled + " frames labeled.";
  clearBtn.disabled = !boxes[index];
  trainBtn.disabled = !labelInput.value.trim() || labeled === 0;
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
  if (!session || !boxes[index]) return;
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
  if (!session || !session.keyframe_count || viewMode !== "train" || !frameImage) return;
  if (ev.button !== undefined && ev.button !== 0) return;
  clearError();
  try { stage.setPointerCapture(ev.pointerId); } catch (err) { /* synthetic events have no capture */ }
  const [x, y] = pointerToImage(ev);
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
  try {
    const data = await postJSON("/api/sessions/" + session.id + "/box", { index, x1, y1, x2, y2 });
    boxes = data.boxes;
    tapHint.classList.remove("on");
    describeFrame();
    paintStage();
    loadMask(index);
    if (labelInput.value.trim()) labelInput.focus();
  } catch (err) {
    showError(err.message);
  }
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
  if (!session) return;
  clearError();
  const index = Number(scrub.value);
  try {
    const data = await postJSON("/api/sessions/" + session.id + "/box", { index, clear: true });
    boxes = data.boxes;
    masks[index] = null;
    describeFrame();
    paintStage();
    if (!boxes.some(Boolean)) tapHint.classList.add("on");
  } catch (err) {
    showError(err.message);
  }
});

labelInput.addEventListener("input", () => {
  trainBtn.disabled = !labelInput.value.trim() || !boxes.some(Boolean);
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
      refreshModel();
      progressEl.textContent += " Switch to Inference to run this model.";
    }
    if (data.status === "error") {
      clearInterval(timerId);
      showError(data.message);
      trainBtn.disabled = false;
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
  resetModelBtn.disabled = !data.can_reset;
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
        self.active_weights = model
        self.weight_stack: list[str] = []
        latest = self._newest_weights()
        if latest and Path(latest).resolve() != Path(model).resolve():
            self.weight_stack.append(model)
            self.active_weights = latest
        self._yolo = None
        self._yolo_path: str | None = None
        self.recorder = CameraRecorder()

    def _newest_weights(self) -> str | None:
        if not self.work_root.is_dir():
            return None
        found = [
            path
            for path in self.work_root.rglob("best.pt")
            if path.is_file() and path.parent.name == "weights"
        ]
        if not found:
            return None
        return str(max(found, key=lambda path: path.stat().st_mtime))

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

    def model_status(self) -> dict:
        previous = self.weight_stack[-1] if self.weight_stack else None
        return {
            "active": self.active_weights,
            "active_name": _weight_label(self.active_weights),
            "previous": previous,
            "previous_name": None if previous is None else _weight_label(previous),
            "can_reset": previous is not None,
        }

    def reset_model(self) -> dict:
        with self.torch_lock:
            if not self.weight_stack:
                raise ApiError(409, "No previous model to restore")
            self.active_weights = self.weight_stack.pop()
            self._yolo = None
            self._yolo_path = None
        return self.model_status()

    def open_frames(self, frames: list[np.ndarray], fps: float) -> Session:
        bundle = process_frames(frames, fps)
        session_id = uuid.uuid4().hex[:12]
        directory = self.work_root / "sessions" / session_id
        saved = save_capture(bundle, directory)
        session = Session(id=session_id, bundle=bundle, directory=directory, saved=saved)
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
            "negative_label_paths": [str(path) for path in session.saved.negative_label_paths],
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
        current = None if index is None or stored[index] is None else list(stored[index])
        return {
            "index": index,
            "bbox": current,
            "boxes": boxes,
            "labeled_count": sum(box is not None for box in stored),
        }

    def set_box(self, session: Session, index: int, x1: int, y1: int, x2: int, y2: int) -> dict:
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
        frame = session.bundle.keyframes[index]
        try:
            session.masks[index] = object_mask(frame, session.boxes[index])
        except Exception:
            fallback = np.zeros((height, width), np.uint8)
            fallback[top : bottom + 1, left : right + 1] = 1
            session.masks[index] = fallback
        return self.box_state(session, index)

    def clear_box(self, session: Session, index: int) -> dict:
        self._require_frame(session, index)
        session.boxes[index] = None
        session.masks[index] = None
        return self.box_state(session, index)

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
            raise ApiError(409, "Draw a box on at least one frame before training")
        dest = session.directory / "labeled"
        if dest.exists():
            shutil.rmtree(dest)
        image_dir = dest / "images"
        label_dir = dest / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        width, height = session.bundle.width, session.bundle.height
        cv2 = _cv2()
        masks = session.masks or [None] * len(session.boxes)
        for path, box, mask in zip(session.saved.keyframe_paths, session.boxes, masks, strict=True):
            if box is None:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            if mask is None or mask.shape[:2] != image.shape[:2]:
                mask = object_mask(image, box)
            seed = zlib.adler32(path.stem.encode("utf-8")) & 0xFFFFFFFF
            views = training_views(image, box, mask, rng=np.random.default_rng(seed))
            for offset, (view, tight) in enumerate(views):
                stem = path.stem if offset == 0 else f"{path.stem}_bg{offset - 1:02d}"
                cv2.imwrite(str(image_dir / f"{stem}.jpg"), view, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                write_yolo_label(
                    label_dir / f"{stem}.txt",
                    0,
                    xyxy_to_yolo(*tight, width, height),
                )
            gone = scene_without_object(image, mask)
            gone_stem = f"{path.stem}_gone"
            cv2.imwrite(str(image_dir / f"{gone_stem}.jpg"), gone, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            write_empty_label(label_dir / f"{gone_stem}.txt")
        for image_path, label_path in zip(
            session.saved.negative_paths,
            session.saved.negative_label_paths,
            strict=True,
        ):
            shutil.copyfile(image_path, image_dir / image_path.name)
            shutil.copyfile(label_path, label_dir / label_path.name)
        return dest

    def start_training(self, session: Session, label: str) -> dict:
        if not session.boxes or not any(box is not None for box in session.boxes):
            raise ApiError(409, "Draw a box on at least one frame before training")
        cleaned = clean_label(label)
        session.label = cleaned
        session.progress = {
            "status": "running",
            "epoch": 0,
            "epochs": LABELED_EPOCHS,
            "fraction": 0.0,
            "message": f"Segmenting {cleaned}, then training it without the background.",
            "remaining_s": None,
            "weights": None,
            "report": None,
        }
        thread = threading.Thread(target=self._train, args=(session, cleaned), daemon=True)
        thread.start()
        return session.progress

    def _train(self, session: Session, label: str) -> None:
        try:
            labeled = self.export_labeled_set(session)
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
                result = train_labeled_class(
                    labeled,
                    label,
                    epochs=LABELED_EPOCHS,
                    model=self.model,
                    device=self.device,
                    project=session.directory / "run",
                    imgsz=self.imgsz,
                    batch=min(batch, 4),
                    progress_callback=progress,
                )
            report = None
            if result.weights and result.weights != self.active_weights:
                with self.torch_lock:
                    self.weight_stack.append(self.active_weights)
                    self.active_weights = result.weights
                    self._yolo = None
                    self._yolo_path = None
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
        import time

        from ultralytics import YOLO

        try:
            with self.torch_lock:
                if self._yolo is None or self._yolo_path != self.active_weights:
                    self._yolo = YOLO(self.active_weights)
                    self._yolo_path = self.active_weights
                started = time.perf_counter()
                result = self._yolo.predict(
                    image, device=self.device, verbose=False, conf=0.15, imgsz=self.imgsz
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
            if path.endswith("/progress") and path.startswith("/api/sessions/"):
                session = self._session_from(path.split("/")[3])
                self._send_json(200, session.progress)
                return
            parts = path.strip("/").split("/")
            if len(parts) == 5 and parts[:2] == ["api", "sessions"] and parts[3] == "masks":
                session = self._session_from(parts[2])
                name = parts[4]
                if not name.endswith(".png"):
                    raise ApiError(404, "mask not found")
                body = app.mask_png(session, int(name[:-4]))
                self._send(200, body, "image/png")
                return
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
            if path == "/api/model/reset":
                self._send_json(200, app.reset_model())
                return
            if path == "/api/infer":
                image = _decode_image(self._body())
                self._send_json(200, app.infer(image))
                return
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "box":
                session = self._session_from(parts[2])
                payload = self._json()
                index = int(payload["index"])
                if payload.get("clear"):
                    self._send_json(200, app.clear_box(session, index))
                    return
                self._send_json(
                    200,
                    app.set_box(
                        session,
                        index,
                        int(payload["x1"]),
                        int(payload["y1"]),
                        int(payload["x2"]),
                        int(payload["y2"]),
                    ),
                )
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
