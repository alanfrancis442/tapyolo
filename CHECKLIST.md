# TapYOLO checklist

**Current focus:** Pipeline is in place. ONNX weights and the ≥90% mAP check are still open.

Status: `[ ]` not started, `[~]` in progress, `[x]` done.

## Phase 1 — Capture & Keyframe Sampler

- [x] Record video, perform SSIM frame filtering, and output 20–30 diverse image frames to disk.
- [x] SSIM threshold is 0.88. If that pass is outside the 20–30 band, fall back to every 6th frame (and resample into the band when the clip still has at least 20 content frames).
- [x] Tag 3–5 frames from the last ~1 second as background samples with empty annotation files.
  - The exporter writes 4 frames from that tail. Pytest covers a 3.0s / 90-frame clip.

## Phase 2 — Tap-to-Box Integration

- [x] A tap coordinate on a keyframe produces a binary mask and the tight box \(x_{min}, y_{min}, x_{max}, y_{max}\).
- [ ] Tap coordinate on the UI produces an exact segmentation mask via MobileSAM in <50 ms.
  - ONNX MobileSAM encoder/decoder path is implemented. It runs only when weights are on disk. Latency is measured with a clock, not filled in.
- [x] Without weights, a deterministic flood-fill / GrabCut fallback still returns a mask and box.

## Phase 3 — Temporal Track Propagation

- [x] Propagate the tapped box backward from the tapped keyframe to frame 0 and forward to the end of the harvested keyframes.
- [x] Forward/backward ByteTrack correctly labels >90% of sampled keyframes without bounding-box drift.
  - Measured on a synthetic 15-frame moving box: every frame stayed within 8 px of the true center, and `supervision.ByteTrack` held one track id (`minimum_matching_threshold=0.9`, a cost ceiling). A camera recording was not scored.
- [x] Write YOLO-normalized labels: `class_id x_center y_center width height`, using the user-supplied label string.

## Phase 4 — Head-Freezing Trainer

- [x] Fine-tune YOLO11n or YOLOv8n with `freeze=10` (backbone layers 0–9).
  - A 1-epoch CUDA smoke on this RTX 4060 logged freezing `model.0` through `model.9`.
- [x] Short run: 10–15 epochs (default 12), `lr0=0.001`. A test can call the same function with a smaller epoch budget.
  - Optimizer is AdamW. Ultralytics `optimizer=auto` replaces `lr0`.
- [x] Augmentations: mosaic p=0.5, HSV (h=0.015, s=0.7, v=0.4), scale ±50%, rotation ±10°. `close_mosaic=0` so mosaic stays on during the short run.
- [x] Composite batch at μ=0.70 novel / 0.30 anchor.
  - Ultralytics applies one detection loss. The loader is filled at that image ratio. The smoke mix fraction was 0.706.
- [x] Rehearsal buffer of about 25 multi-class anchor images.
  - The bundled set is generated locally (person, bicycle, car, cup, chair). `--anchors` uses a local folder when you supply one. COCO is not downloaded.
- [ ] Automated fine-tuning completes in <25 seconds on an RTX 3060.
  - Not measured. A 1-epoch run at imgsz 64 / batch 2 on this RTX 4060 finished the training loop in about 2 seconds and is not that budget.

## Phase 5 — Anti-Forgetting Validation

- [x] Run the tuned model on the anchor set and on the new class, and report whether anchor detection is retained.
- [ ] Confirm retained accuracy of ≥90% baseline mAP on bundled anchor images while identifying the new class.
  - After the 1-epoch imgsz-64 smoke, measured anchor mAP50 was 0.0 and novel mAP50 was 0.0 (`meets_90_map` false). That run is not the 10–15 epoch product budget.

## Cross-cutting — uv project and dependencies (§6)

- [x] `pyproject.toml`: name `tapyolo`, `requires-python >=3.11,<3.13`, the PRD dependency list, dev extras, and the `pytorch-cu124` uv index for torch/torchvision.
- [x] README documents `uv venv`, `uv sync`, and `uv run python -m tapyolo.main --model yolo11n.pt --device cuda:0`.
- [x] `--device cuda:0` resolves to `cpu` when CUDA is unavailable, and to `mps` when MPS is the available accelerator. The pyproject indexes stay on cu124.
  - This machine has CUDA. `resolve_device("cuda:0")` is `cuda:0` (torch 2.6.0+cu124).

## Cross-cutting — UI flow (§7)

- [x] Record → Stop → freeze on a keyframe → tap the target → mask and box feedback → scrubber of tracked boxes → label text + Train Now → progress → live camera inference with the tuned class.
  - Verified over HTTP (upload, keyframes, tap, tracks, page controls). No in-app browser tool was available, so the page was not clicked through visually.
- [x] CLI entry: `python -m tapyolo.main --model yolo11n.pt --device cuda:0`.

## Cross-cutting — hardware budgets (§5, §8)

- [x] CUDA training batch defaults to 8 at 640 (the PRD ~1.4 GB YOLO batch). CPU/MPS batch defaults to 2.
- [ ] Segmenter ≤50 ms, tracker ≤3 ms/frame, training ≤25 s, inference ≥60 FPS. Recorded only from a real measurement.
  - The HTTP tap test records `elapsed_ms` from the clock. Those PRD ceilings were not checked against it.
- [~] VRAM ceiling targeted by the batch setting is ≤2.2 GB. Peak allocation on a 640 / batch-8 run was not logged.
  - The imgsz-64 smoke reported `GPU_mem` 0.141G.

## Cross-cutting — tests

- [x] pytest: SSIM/stride sampler bounds (20–30 when enough frames exist).
- [x] pytest: bbox normalization.
- [x] pytest: YOLO label writing.
- [x] pytest: empty negative labels.
- [x] pytest: freeze config.
- [x] pytest: tracker propagation on a synthetic moving box.
- [x] Tests do not download multi-GB weights.
  - `uv run pytest -q`: 28 passed.

## Blockers

- MobileSAM / FastSAM ONNX weights are not in the workspace. The ONNX segmenter stays unused until `weights/mobile_sam_encoder.onnx` and `weights/mobile_sam_decoder.onnx` (or `TAPYOLO_MOBILESAM_ENCODER` / `TAPYOLO_MOBILESAM_DECODER`) exist.
- ≥90% anchor mAP is not met on the only training run so far (1 epoch, imgsz 64, synthetic anchors and a synthetic novel class).
- The <25 s RTX 3060 training budget and the <50 ms MobileSAM tap budget were not measured.
- cu124 sync succeeded here (RTX 4060 Laptop, 8 GB). `yolo11n.pt` (5.4 MB) was downloaded into the repo root on the smoke run and is gitignored.
