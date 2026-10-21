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
  document.getElementById("view").classList.add("has-media");
  preview.classList.add("on");
  stage.classList.remove("on");
  overlay.classList.remove("on");
}

function showStage() {
  preview.classList.remove("on");
  stage.classList.add("on");
  overlay.classList.remove("on");
  document.getElementById("view").classList.toggle("has-media", Boolean(session));
}

function showLiveOverlay() {
  document.getElementById("view").classList.add("has-media");
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
  document.getElementById("timelineCount").textContent = data.keyframe_count + " keyframes";
  tapHint.textContent = "Tap the object once. Refine with foreground/background points, or draw a correction box.";
  tapHint.classList.add("on");
  describeFrame();
  if (data.warning) meta.textContent += " " + data.warning;
  trainBtn.disabled = true;
  clearBtn.disabled = true;
  if (data.keyframe_count) drawFrame(0);
}

function describeFrame() {
  const frameStatus = document.getElementById("frameStatus");
  if (!session) {
    frameStatus.textContent = "Waiting for a video";
    document.getElementById("timelineCount").textContent = "No frames yet";
    return;
  }
  const index = Number(scrub.value);
  const labeled = boxes.filter(Boolean).length;
  frameStatus.textContent = "Frame " + (index + 1) + " of " + session.keyframe_count + " · " + labeled + " labeled";
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
