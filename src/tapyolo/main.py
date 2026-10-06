"""CLI entry: ``python -m tapyolo.main --model yolo11n.pt --device cuda:0``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tapyolo.capture import save_capture
from tapyolo.device import batch_for_device, resolve_device
from tapyolo.labels import clean_label, write_tracked_labels
from tapyolo.pipeline import load_video, tap_keyframes
from tapyolo.segment import load_segmenter
from tapyolo.train import DEFAULT_EPOCHS, train_detection_head
from tapyolo.ui import TapYoloApp, serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive single-tap video labeling and ultra-fast YOLO head adaptation",
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--tap", help="x,y pixel coordinate on a harvested keyframe")
    parser.add_argument("--keyframe", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/tapyolo"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=0)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    print(f"Resolved device: {device} (requested {args.device}).")
    if args.epochs < 10 or args.epochs > 15:
        print(f"Epoch budget is {args.epochs}. The short adaptation run uses 10-15 epochs.")

    if args.video is None:
        app = TapYoloApp(
            model=args.model,
            requested_device=args.device,
            work_root=args.output,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
        )
        app.device = device
        serve(app, args.host, args.port)
        return 0

    bundle = load_video(args.video)
    saved = save_capture(bundle, args.output)
    summary: dict = {
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
        "output": str(args.output),
    }
    if args.tap:
        x_text, y_text = args.tap.split(",")
        choice = load_segmenter()
        tapped = tap_keyframes(bundle.keyframes, args.keyframe, int(x_text), int(y_text), choice.segmenter)
        summary["segmenter"] = tapped["backend"]
        summary["segment_detail"] = tapped["detail"]
        summary["segment_ms"] = tapped["elapsed_ms"]
        summary["tracker"] = tapped["tracker"]
        summary["track_ms"] = tapped["track_ms"]
        if args.label:
            label = clean_label(args.label)
            boxes = [None if track["bbox"] is None else tuple(track["bbox"]) for track in tapped["tracks"]]
            stems = [path.stem for path in saved.keyframe_paths]
            write_tracked_labels(args.output / "labels", stems, boxes, 0, bundle.width, bundle.height)
            (args.output / "classes.txt").write_text(label + "\n", encoding="utf-8")
            summary["label"] = label
            if args.train:
                result = train_detection_head(
                    args.output,
                    label,
                    epochs=args.epochs,
                    model=args.model,
                    device=device,
                    project=args.output / "run",
                    imgsz=args.imgsz,
                    batch=batch_for_device(device, args.batch or None),
                    anchor_dir=args.anchors,
                )
                summary["weights"] = result.weights
                summary["mix_fraction"] = result.mix_fraction
                summary["train_note"] = result.note
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
