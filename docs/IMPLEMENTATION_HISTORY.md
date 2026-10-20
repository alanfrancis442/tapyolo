# TapYOLO: investigation, implementation history, and current behavior

Date: 2026-10-20.

This document records the work from the initial repository review through the restoration of automatic video annotation. It separates implemented behavior, observed test results, and product targets that remain unverified.

## 1. Original objective and starting point

The intended workflow was to record a short video, tap an object once, automatically label it across useful frames, and quickly adapt a YOLO detector to recognize the new class. The original specifications emphasized short training runs, a frozen backbone, rehearsal of old classes, and measured retention of existing detection accuracy.

The repository already contained most of the necessary components: capture and video decoding, SSIM keyframe sampling, point segmentation, box propagation, YOLO label export, fine-tuning, evaluation utilities, a local HTTP server, and an embedded browser interface.

However, the implementation had evolved into two different workflows:

| Area | State at the initial review |
| --- | --- |
| CLI | Tap-based segmentation, template-based propagation, and an optional 12-epoch rehearsal training run |
| Browser | Manually drawn boxes, object/background augmentation, and single-class training for 40 epochs |
| Documentation | Mostly described the earlier single-tap browser workflow |
| Segmentation weights | MobileSAM ONNX support existed, but the required weights were absent |
| Rehearsal | Generated colored shapes labeled with familiar class names |
| Validation | Training and validation reused the same image directories |
| Model management | Latest-weight loading and rollback existed, but there was no saved-model selector or deletion control |

The switch to manually drawn boxes was a practical response to unreliable segmentation. The subsequent investigation showed that fixing the initial segmentation alone would not be enough to restore reliable automation across a video.

## 2. Why the initial segmentation was inconsistent

The project could load MobileSAM ONNX encoder/decoder files if they were present. They were not present in the workspace, so taps normally used OpenCV flood fill and GrabCut instead.

Flood fill groups pixels using local color similarity. GrabCut separates foreground and background using appearance within an initialized region. These methods can work well for a contrasting object against a simple background, but similar object/background colors, lighting, hands, and loose prompts can change their results substantially.

This explains how the same interaction could sometimes select the object accurately and sometimes select only a fragment or include background. The application had a learned-model integration point, but the learned model was not the backend actually doing the work.

### Changes to point segmentation

- Installed official MobileSAM PyTorch weights locally and added a native backend using the installed Ultralytics integration.
- Kept the optional ONNX backend and corrected its normalized padding, low-resolution mask unpadding, and prompt padding.
- Added positive foreground points and negative background points for refining one selection.
- Added candidate-mask checks: the mask must agree with the supplied points and must not be empty or cover nearly the whole scene.
- Cached the image embedding so refinement on the same image can avoid another encoder pass.
- Removed the last-resort behavior that fabricated a small rectangle when segmentation could not isolate an object.
- Kept the OpenCV fallback explicitly identified when learned weights cannot load.
- Kept drawn rectangles as authoritative corrections rather than shrinking them again using the same color heuristics the user was correcting.

An actual MobileSAM smoke check produced a mask for a single positive point and for a positive/negative refinement. This established that the learned backend was running; it did not establish accuracy on every possible object or background.

## 3. Why the first repair still required too much manual work

After learned segmentation was restored, the supplied screenshots showed a useful distinction: the initial remote-control mask was good, but later boxes moved onto the hand and wall as the remote rotated.

That was a propagation failure. The tracker still compared later frames with a fixed grayscale image patch taken from the first selection. It searched nearby locations and a small set of scales, but it did not learn the remote's changing appearance or remember it as a video object.

The previous capture representation also retained only sampled keyframes for tracking. Skipping the intervening motion can turn a gradual rotation into a large appearance change between the frames the tracker sees.

Stopping propagation when a match became weak helped prevent invented labels, but it did not solve cases where the wrong hand/background patch produced a plausible correlation score. More manual corrections were still needed than the intended workflow allowed.

## 4. Replacing fixed-template propagation with video segmentation

The normal propagation path now uses SAM 2.1 tiny and its video memory. MobileSAM still provides the initial selected mask. SAM 2.1 follows that selected object through the video, and each keyframe's box is calculated from its resulting mask.

The three learned-model roles are distinct:

| Model | Responsibility |
| --- | --- |
| MobileSAM | Select the object from the user's tap and optional refinement points |
| SAM 2.1 | Follow the selected object through changing video frames |
| YOLO | Learn the resulting class and detect it later during live inference |

### Consecutive frames and sampled keyframes

New captures retain consecutive content frames for tracking, bounded to a maximum long side of 640 pixels. They also save a `tracking.avi` file. The original keyframes remain available at capture resolution for review and training.

SSIM sampling still targets 20–30 useful keyframes. The change is that temporal tracking sees the intervening frames rather than jumping directly between only those sampled images.

The tracker maps the video masks back to the selected keyframe indices and resizes them to the original annotation dimensions.

### Tracking behavior

- Propagation runs forward and backward from the selected frame, using separate directional video passes.
- Later manual selections can condition the tracker when it reaches those frames.
- Manual corrections and excluded frames survive subsequent propagation.
- Propagated masks are retained and displayed; later frames no longer receive only an unverified rectangle.
- Empty, scene-sized, implausibly resized, or abruptly displaced masks are excluded rather than turned into guessed labels.
- The browser polls annotation progress so a longer video pass does not look like an unresponsive selection.
- Tracking state is cleared between passes/sessions to avoid carrying one clip's object memory into another clip.
- A mask-input channel dimension required by the installed SAM 2 integration is supplied explicitly; the real-model smoke test caught this integration issue.

The older template/ByteTrack path remains an explicitly reported fallback when SAM 2 weights are missing. The normal installed configuration should display `tracker sam2-video` in the browser status.

### Evidence from the remote-control recording

The saved recording contained 30 sampled keyframes. Starting from one MobileSAM selection, the video tracker returned masks for all 30. Seven views were inspected visually, covering upright, horizontal, back, and button-side orientations. Those inspected views stayed on the remote rather than reproducing the wall drift shown in the screenshots.

This test used the older saved keyframes because that capture predated consecutive-frame retention. New recordings provide the additional intervening frames.

Mask coverage is not a quantitative segmentation accuracy measurement: 30 returned masks does not mean 100% pixel accuracy. No manually labeled ground-truth evaluation was performed. A separate real-model smoke test also exercised backward tracking from an interior frame and a later correction, returning masks for all six test frames and preserving the supplied masks.

SAM 2 video processing costs more compute than the previous patch matcher. Its latency and memory requirements need their own measurements; the original 3 ms/frame tracker target is not claimed for this implementation.

## 5. Training changes

### Shared production training path

The browser now uses the rehearsal-aware adaptation trainer also used by the CLI. Its production defaults are 12 epochs, `freeze=10`, AdamW, and `lr0=0.001`. The configured epoch count, image size, and batch size are honored. Setting the nominal batch size to the actual batch size prevents Ultralytics from scaling away the intended learning rate.

The former 40-epoch single-class helper remains in the code for compatibility; it is no longer the browser's production training path.

### Preserve learned classification layers when adding classes

A training smoke test exposed an additional transfer issue. YOLO's default classification hidden width can depend on the number of classes. Expanding the class count can therefore rebuild hidden classification layers at a different width, leaving some pretrained tensors unmatched.

The new adaptation trainer copies the existing detector architecture, preserves those learned hidden layers, and replaces only the final class-output convolutions. Existing output rows are copied by class name, and new class rows are initialized separately. Class counts and model metadata are updated accordingly.

An actual checkpoint check found a maximum difference of `0.0` between the old-class outputs before and immediately after expansion, before training. This verifies initial preservation of those outputs. Training can still change the detector afterward, so rehearsal and evaluation remain necessary.

This expansion currently supports standard YOLO11/YOLOv8 detection heads. Unsupported/end-to-end head layouts are rejected explicitly.

### Real rehearsal data

Production training now uses real COCO128 photographs and their original labels. By default, 25 originals are used for rehearsal training and the other originals are held out. A user-supplied rehearsal directory is supported through `--anchors`, with `classes.txt` specifying the class-id order.

Synthetic colored-shape anchors remain available as test/demo fixtures, but they are no longer the implicit production rehearsal dataset.

Custom/rehearsal images are mixed near 70/30 by repeating either side as needed. This is an image-loader mixture with one Ultralytics detection loss, not two independently weighted loss functions.

### Remember earlier custom classes

New checkpoints save provenance describing the custom datasets and class names used to train them. Training from a selected saved model includes examples for its earlier custom classes as well as the new class. Compatible saved models provide the starting checkpoint; legacy single-class models can recover their original dataset from their saved training configuration.

Selecting the base model starts a fresh collection of custom classes. Missing replay data or unsupported legacy provenance produces an explicit error rather than silently claiming the earlier classes were retained.

Repeated training creates a new run and a new export rather than overwriting data an older checkpoint may still reference.

## 6. Validation and background-label correctness

### Separate training and validation originals

Previously, the same image directories were used for training and validation. A high score could therefore reflect memorization of the training examples.

The new dataset builder separates original frame groups. An original image, its background variations, and its object-removed derivative all belong to one group. No derivative of a held-out original appears in the training split. Validation uses original images only.

At least two labeled original keyframes are required so there can be separate training and validation examples. Replay sources receive the same grouping treatment.

These held-out frames still come from the same recordings. Separate recordings and environments are needed to assess generalization beyond those clips.

### Do not confuse unknown frames with background

A lost track means the object location is unknown; it does not establish that the object is absent. Such frames are skipped during training.

Likewise, the final second is only a candidate background tail. The browser now requires the checkbox confirming that the object was absent during that interval. The CLI has the equivalent `--background-tail` flag. Without confirmation, those tail frames are excluded from training.

### Measure retention against an actual baseline

The app evaluates the configured base model and tuned model on the same held-out anchor photographs, aligning predictions by class name. It reports:

- Baseline and tuned anchor mAP50.
- Relative retention: tuned anchor mAP50 divided by measured baseline anchor mAP50.
- Whether relative retention reaches 90%.
- Whether absolute anchor mAP50 reaches 0.90, as a separate condition.
- Held-out custom-class mAP50, recall, and per-class AP.

For example, a baseline mAP50 of 0.60 and tuned score of 0.54 represent 90% relative retention, not absolute mAP50 of 0.90. When the baseline score is zero, relative retention is unmeasurable.

The report is saved beside the trained weights. It reports the target result; it is not a hard gate that blocks activation of every model falling below the target. The user can inspect the result and select or restore another model.

## 7. Model selection, rollback, and deletion

The browser now lists trained checkpoints under the configured output root and provides **Use selected model** and **Delete selected trained model** controls.

Deletion requires an in-browser confirmation and removes that selected run's checkpoint files, including `best.pt` and `last.pt`. Captures, labels, exported datasets, and training logs are retained. Keeping the datasets also preserves replay sources that a newer model may reference.

If the active model is deleted, the application switches to a surviving previous model or the base model and clears the cached inference model. The base model is protected. The API resolves catalog ids rather than accepting arbitrary deletion paths, and symlinked checkpoints are excluded.

The active selection persists across restarts. The full in-memory rollback stack is not persisted as a complete historical undo log. Model selection, deletion, and duplicate training jobs are blocked while training is running.

## 8. Setup, storage, and dependencies

Setup is reproducible with:

```bash
uv sync --frozen --extra dev
uv run python -m tapyolo.setup
uv run python -m tapyolo.main --model yolo11n.pt --device cuda:0
```

The setup command downloads official MobileSAM weights, SAM 2.1 tiny weights, and COCO128 into `weights/`. Approximate download sizes are 39 MB, 75 MB, and 7 MB respectively. Existing installed assets are reused. ONNX and model-path environment overrides remain documented in the README.

The minimum Ultralytics requirement is now `8.4.173`, matching the integration used and tested; the lockfile was updated and checked.

Model weights, recordings, generated datasets, training outputs, and the local graft index are excluded from version control. Source code, tests, setup instructions, and this implementation history are committed instead. The ignore configuration also keeps graft's generated cache out of ordinary source search results.

## 9. Verification and remaining work

### Automated verification

The full suite passed with 49 tests after the video-tracking changes. Coverage includes segmentation fallback behavior, prompt constraints, SAM padding, synthetic tracking loss, manual correction/exclusion preservation, grouped splits, replay class mapping, classification-head expansion, training configuration, evaluation ratios, HTTP routes, confined checkpoint deletion, consecutive-frame retention, bidirectional video mapping, and propagated masks.

The browser JavaScript passed `node --check`; Python compilation, import/undefined-name lint checks, and Git whitespace checks also passed. The dependency lockfile passed `uv lock --check --offline`.

Tests use deterministic synthetic fixtures and fake predictors/trainers where appropriate; they do not download or run all learned models as part of every test invocation. Real-model checks were performed separately.

### Training smoke result and its limits

Actual YOLO training plus baseline/tuned evaluation completed on CPU at one epoch, image size 64, batch size 2, with a deliberately tiny rehearsal subset. After preserving the hidden head layers, the run reported:

| Measurement | Tiny smoke-run result |
| --- | --- |
| Baseline anchor mAP50 | 0.25 |
| Tuned anchor mAP50 | Approximately 0.0209 |
| Relative anchor retention | Approximately 8.37%; target not met |
| Custom-class mAP50 | 1.0 on one held-out synthetic original |
| Held-out anchors | Four images |

These numbers verify that training and reporting execute and that the report correctly records a failed retention target. They are not representative accuracy claims. In particular, a perfect score on one synthetic validation image is not evidence of useful real-world novel-class accuracy.

### Outstanding product validation

- Evaluate segmentation and tracking on varied real objects, occlusions, backgrounds, lighting, and independent recordings.
- Verify useful novel-class detection and at least 90% relative base-class retention under the full training configuration.
- Measure MobileSAM latency, SAM 2 video throughput, training duration, peak GPU memory, and end-to-end live inference throughput on the intended hardware.
- Perform a visual browser/camera acceptance check; HTTP tests and JavaScript syntax checks do not replace it.

The original latency, training-time, VRAM, and inference-FPS budgets remain unverified. Introducing video-memory tracking changes the cost of the annotation stage and should be reflected in future hardware measurements.

## 10. Main implementation files

| File | Responsibility |
| --- | --- |
| `src/tapyolo/capture.py` | Capture bundles, sampling input, consecutive-frame retention, and saved tracking video |
| `src/tapyolo/segment.py` | MobileSAM/ONNX point segmentation, refinement, mask checks, and image augmentation |
| `src/tapyolo/video_tracker.py` | SAM 2.1 video memory, directional passes, later corrections, mask checks, and frame mapping |
| `src/tapyolo/tracker.py` | Explicit legacy template/ByteTrack fallback, with unreliable-match stopping |
| `src/tapyolo/pipeline.py` | Shared connection between tap selection and propagation |
| `src/tapyolo/adaptation.py` | Preserve detector hidden layers while expanding class outputs |
| `src/tapyolo/train.py` | Training settings, class namespaces, grouped train/validation datasets, and rehearsal mixing |
| `src/tapyolo/validate.py` | Held-out detection scoring and baseline-relative retention reports |
| `src/tapyolo/models.py` | Confined checkpoint catalog and replay provenance recovery |
| `src/tapyolo/ui.py` | Browser controls, sessions, annotation progress, training jobs, inference, and model management |
| `src/tapyolo/main.py` | CLI/server entry point and configuration wiring |
| `src/tapyolo/setup.py` | Reproducible downloads and real rehearsal-data preparation |
| `tests/` | Unit, integration, HTTP, and regression coverage |

The README describes how to use the current application. CHECKLIST.md distinguishes completed implementation from outstanding measurements. This document preserves the reasoning and evidence behind the changes.
