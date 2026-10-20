# TapYOLO

Label an object in a short video with a tap, review its tracked boxes, and adapt a YOLO detector locally.

For the investigation history, design decisions, complete change summary, and measured results, see [Implementation history](docs/IMPLEMENTATION_HISTORY.md).

## Setup

Python 3.11–3.12 and `uv` are required. Torch/torchvision use the CUDA 12.4 index.

```bash
uv sync --frozen --extra dev
uv run python -m tapyolo.setup
```

Setup downloads official MobileSAM weights (~39 MB), SAM 2.1 tiny video weights (~75 MB), and COCO128 (~7 MB) into the gitignored `weights/` directory. It is safe to rerun. Production training samples 25 real rehearsal images, holding the other originals out for validation. Synthetic anchors remain available only as test/demo fixtures.

```bash
uv run python -m tapyolo.main --model yolo11n.pt --device cuda:0
```

Open `http://127.0.0.1:7860`. CUDA falls back to MPS or CPU when unavailable. CPU segmentation and training can be substantially slower; the project does not claim the GPU timing budgets on CPU. The default YOLO checkpoint can download on first use.

## Label and train

1. Record a 3–8 second clip, or open a video. Move around the object to capture different views. If you want background examples, remove the object during the final second.
2. Tap the target on a keyframe. MobileSAM generates the initial mask. SAM 2.1 then follows that selected object through consecutive video frames in both directions, using video memory to handle rotation and changing views. Boxes are recomputed from the resulting masks on each keyframe. The UI displays tracking progress.
3. Review the scrubber. Use **Add foreground point** for missing object parts and **Exclude background point** for unwanted regions. **Tap object (new selection)** replaces that frame's points. Cached image features make repeated refinement cheaper.
4. Use **Draw correction box** when points cannot isolate the object, then **Propagate this correction**. A drawn box is authoritative: color heuristics do not shrink it. Manual selections and cleared frames survive later propagation. Later corrections also condition the video tracker as it passes those frames. **Clear this frame** excludes it from training.
5. Check **The object is absent during the final second** only when that is true. Unconfirmed tail frames are excluded from training. Tracking stops at unreliable matches; missing tracks are excluded rather than used as negative examples.
6. Enter the class name and choose **Train Now**. At least two labeled keyframes are needed to separate training and validation. Both the browser and CLI default to 12 epochs, `freeze=10`, AdamW, and `lr0=0.001`; `--epochs`, `--imgsz`, and `--batch` are honored.
7. Switch to **Inference** to use the trained detector. Select a saved model with **Use selected model**, or use **Reset to previous model**.

The goal is one initial selection followed by automatic video annotation; ordinary rotation should not require drawing each frame. Masks and tracks still need review. Occlusion, similar objects, abrupt cuts, and ambiguous taps can still defeat segmentation or tracking. Empty, scene-sized, or discontinuous masks are excluded instead of becoming guessed labels. OpenCV is an explicitly reported fallback when learned weights cannot load; it is less reliable. A failed learned selection asks for correction instead of silently substituting a guessed rectangle.

## Training and retention

The trainer preserves the checkpoint's hidden classification layers and expands only its final class-output convolutions, copying existing rows by class name. This avoids losing pretrained hidden weights when YOLO's default head width changes as classes are added.

The loader mixes custom examples and real base-class rehearsal images near 70/30. Original labeled frames, object cutouts on varied backgrounds, and object-removed negatives supply custom training examples. Every derivative of a source frame stays in the same split. Validation contains original images only; it never evaluates a synthetic copy of a training frame.

New models retain replay provenance. Training from a selected model includes its earlier custom classes and their saved examples. Selecting the base model starts a fresh collection of custom classes. Legacy single-class UI models can recover their original dataset from the saved training configuration; models with missing provenance require another selection. Repeated training creates a new run and preserves earlier exports.

After training, the app scores the tuned model and base model on the same held-out real anchor photographs. It reports:

- Anchor mAP50 before and after adaptation.
- Retention as tuned anchor mAP50 / measured baseline mAP50, with a target of at least 90%.
- Held-out custom-class mAP50 and per-class scores in the saved report.

A zero baseline makes retention unmeasurable. Absolute mAP50 ≥0.90 is reported separately from 90% retention. Video validation comes from the same source clips, so these scores are not independent-scene generalization evidence. Training does not guarantee accuracy or the hardware budgets; review the measured report and test on another recording.

## Saved models

**Saved models** lists existing checkpoints under the output directory. **Delete selected trained model** removes that run's checkpoint files, including `best.pt` and `last.pt`, after an in-browser confirmation. Captures, labels, datasets, and training logs are kept. Deleting the active model switches to a surviving previous model or the base model. The active selection persists across restarts. The base model cannot be deleted, and model changes/deletion are blocked during training. Existing models are not deleted automatically.

## CLI and configuration

```bash
uv run python -m tapyolo.main --model yolo11n.pt --device cuda:0 \
  --video clip.mp4 --tap 200,150 --label "torque wrench" \
  --train --background-tail --output runs/clip
```

Use `--background-tail` only if the target is absent during the final second. Omit `--train` to export labels without training. Use a fresh output directory per CLI clip. The browser keeps each capture and training run separately.

- Keyframes: SSIM threshold 0.88; stride-6/even-spacing fallback; 20–30 keyframes when enough content frames exist.
- Point model: `weights/mobile_sam.pt`, or `TAPYOLO_MOBILESAM_WEIGHTS`.
- Video tracker: `weights/sam2.1_t.pt`, or `TAPYOLO_SAM2_WEIGHTS`. Run setup to install it; the status line should say `tracker sam2-video`. Missing weights are explicitly reported as the older template fallback. The video tracker uses more compute than template matching and does not have a measured 3 ms/frame guarantee.
- Each new capture retains consecutive content frames at up to 640 pixels on the long side and saves `tracking.avi`. Sampling still selects 20–30 images for review/training; tracking does not skip the intervening motion frames.
- Optional ONNX backend: `weights/mobile_sam_encoder.onnx` and `weights/mobile_sam_decoder.onnx`, or `TAPYOLO_MOBILESAM_ENCODER` and `TAPYOLO_MOBILESAM_DECODER`. It supports multiple point prompts, cached embeddings, and decoder-padding correction. ONNX takes precedence when a valid pair is present.
- Real rehearsal override: `--anchors path` with flat `images/`, `labels/`, and `classes.txt` containing one class name per line in label-id order. No implicit synthetic rehearsal fallback is used.
- Output: `--output` defaults to `runs/tapyolo`; UI sessions live under `sessions/`.

## Verification

```bash
uv run pytest -q
```

Tests use deterministic fixtures and fake trainers; they do not download weights. Coverage includes point refinement, tracking loss, manual correction preservation, grouped validation splits, class replay, classification-head expansion, training settings, HTTP endpoints, confined model deletion, dense video/keyframe mapping, bidirectional video masks, and later-frame corrections. Learned segmentation and real training also require separate smoke/quality checks with installed weights.

References: [MobileSAM point prompts](https://docs.ultralytics.com/models/mobile-sam/), [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/), [Ultralytics training](https://docs.ultralytics.com/modes/train/), [SAM 2 video segmentation](https://docs.ultralytics.com/models/sam-2/).
