# TapYOLO

Interactive single-tap video labeling and fast YOLO head adaptation.

Record a 3–8 second clip, tap the object once, and the app writes YOLO boxes across the harvested keyframes. Train Now freezes the backbone (`freeze=10`) and fine-tunes the detection head for a short run (default 12 epochs, `lr0=0.001`).

## Setup

The project is managed with `uv`. Python is `>=3.11,<3.13`. Torch and torchvision are indexed from the CUDA 12.4 wheel index in `pyproject.toml`.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync --extra dev
```

After `uv.lock` exists, a later install can use `uv sync --frozen --extra dev`.

### CPU and MPS

`--device cuda:0` is the default request. When `torch.cuda.is_available()` is false, the process uses `mps` if Apple Silicon exposes it, and `cpu` otherwise. The pyproject indexes stay on cu124. A machine without that CUDA build can still run the capture, tap, and label path once the environment has a CPU torch wheel installed beside this file.

## Run

```bash
uv run python -m tapyolo.main --model yolo11n.pt --device cuda:0
```

The UI is at `http://127.0.0.1:7860`.

1. **Record** (browser camera, or the OpenCV camera if the browser has none), then **Stop**. **Open video** accepts an existing file.
2. The view freezes on keyframe 0. Tap the target.
3. The mask and box appear, and the scrubber shows the propagated boxes.
4. Type a label and press **Train Now**. The bar follows epoch progress.
5. Live mode posts camera frames to the tuned weights and draws the new class.

Non-interactive path:

```bash
uv run python -m tapyolo.main --model yolo11n.pt --device cuda:0 \
  --video clip.mp4 --tap 200,150 --label "torque wrench" --train --output runs/clip
```

Training downloads `yolo11n.pt` on first use (a few megabytes). It does not download COCO.

## Pipeline notes

- Keyframes: SSIM below 0.88, targeting 20–30 frames, with a stride-6 fallback.
- The last ~1 second contributes 4 empty-label background images (the PRD asks for 3–5).
- Without MobileSAM ONNX weights, taps use flood fill, then GrabCut. Drop `mobile_sam_encoder.onnx` and `mobile_sam_decoder.onnx` in `weights/`, or set `TAPYOLO_MOBILESAM_ENCODER` and `TAPYOLO_MOBILESAM_DECODER`.
- The tracker proposes boxes with template matching and an IoU/motion gate, backward to frame 0 and forward to the end. Installed `supervision.ByteTrack` confirms the identity when it can hold one track.
- The rehearsal set is 25 generated multi-class images. `--anchors path` replaces it with a local folder (`images/`, `labels/`, optional `classes.txt`).
- Ultralytics trains one loss on a loader mixed at μ=0.70 novel / 0.30 anchor. The optimizer is AdamW so `lr0=0.001` is kept (`optimizer=auto` would replace it).
- After training, anchor mAP50 is measured. The app reports that number. It does not treat 0.90 as met unless the measurement reaches it.

## Tests

```bash
uv run pytest
```

The tests cover the sampler, box normalization, label files, empty negatives, freeze settings, synthetic tracking, and the HTTP record/tap flow. They do not download YOLO weights.
