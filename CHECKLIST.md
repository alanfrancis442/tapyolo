# TapYOLO implementation status

Updated 2026-10-20. This describes current behavior; checked implementation items are not claims that accuracy or latency targets have been met.

## Capture and annotation

- [x] Browser/OpenCV recording and video upload.
- [x] SSIM 0.88, bounded 20–30 keyframes when enough content frames exist.
- [x] Four final-second background candidates, used for training only after user confirmation.
- [x] Learned MobileSAM point segmentation using locally installed official weights.
- [x] Foreground/background refinement, multimask selection, point/area validation, cached image features.
- [x] Optional ONNX backend with correct normalized padding, low-resolution unpadding, prompt padding, and cached embeddings.
- [x] Explicitly identified OpenCV fallback when no learned backend can load.
- [x] SAM 2.1 video-memory propagation from one initial selected mask, in both directions.
- [x] Track consecutive capture frames at bounded resolution; derive keyframe boxes from tracked masks.
- [x] Condition video memory on later manual corrections, retain frame exclusions, and report annotation progress.
- [x] Explicit template/ByteTrack fallback only when video weights are missing.
- [x] Stop propagation at unreliable matches; skip unknown frames in training.
- [x] Box correction, manual-selection preservation, and frame exclusion.
- [ ] Verify segmentation accuracy on varied real objects, lighting, occlusion, and backgrounds.
- [ ] Measure MobileSAM ≤50 ms and tracking ≤3 ms/frame on target GPU hardware.

## Training and evaluation

- [x] Browser and CLI share the default 12-epoch adaptation trainer (`freeze=10`, AdamW, `lr0=0.001`).
- [x] Honor configured epochs, batch size, and image size.
- [x] Preserve hidden class-head weights when expanding final class outputs; align rows by class name.
- [x] Real COCO128 rehearsal: 25 training originals and a separate held-out subset.
- [x] Mix custom/rehearsal images near 70/30, repeating either side when needed.
- [x] Replay previous custom-class examples when training from a selected saved model.
- [x] Preserve old exports and create separate runs for repeated training.
- [x] Group all derivatives of an original in one split; validate only original images.
- [x] Measure base and tuned anchor mAP50 on the same held-out images; report relative retention separately from absolute accuracy.
- [x] Report held-out custom-class scores and persist report/provenance beside weights.
- [ ] Verify ≥90% relative retention and useful novel-class accuracy under the full product training budget on independent recordings.
- [ ] Measure training ≤25 s on RTX 3060, peak VRAM ≤2.2 GB, and end-to-end inference ≥60 FPS.

## Model management

- [x] List and select saved trained models and the base model.
- [x] Delete a selected run's checkpoints from the browser, preserving captures/datasets.
- [x] Protect base model and reject arbitrary paths/symlinks.
- [x] Restore a surviving previous/base model when the active one is deleted.
- [x] Persist active selection across restarts.
- [x] Block model mutation during training; prevent duplicate training jobs.

## Validation limits

- Automated tests use synthetic images and fake trainers, with no downloads.
- Local learned-model smoke: actual MobileSAM produced a mask for a single positive point and for positive/negative refinement.
- Local training smoke: actual YOLO training and held-out baseline/tuned evaluation completed at 1 epoch, image size 64, batch 2, on CPU. This only validates execution; it is not the product accuracy or hardware benchmark. The tiny run did not meet the retention target.
- Held-out video frames are disjoint from training originals but still share their source recordings. Independent recordings are needed for generalization evidence.
- The browser JavaScript is syntax-checked and API flows are exercised over HTTP. A visual camera/browser acceptance check is still needed.

## Video automation follow-up

- Installed official SAM 2.1 tiny weights and made video-memory propagation the normal UI/CLI path.
- On the saved remote-control clip (30 keyframes), one initial MobileSAM selection produced video masks for all 30 frames. Seven sampled views were visually inspected, including upright, horizontal, and front/back rotations. No quantitative ground-truth accuracy claim is made from mask coverage alone.
- Consecutive frames are now saved for new captures; older saved captures contain only their sampled frames.
- Video processing latency and GPU memory limits need fresh measurements; this replaces the fixed-template tracker and changes its compute budget.
