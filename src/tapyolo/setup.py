"""Install the small official segmentation model and real rehearsal dataset."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml

SAM_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt"
VIDEO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt"
COCO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def install_anchors(archive: Path, dest: Path) -> None:
    import ultralytics

    config = Path(ultralytics.__file__).parent / "cfg/datasets/coco128.yaml"
    names = yaml.safe_load(config.read_text())["names"]
    names = (
        [names[index] for index in sorted(names)] if isinstance(names, dict) else names
    )
    for folder in ("images", "labels"):
        (dest / folder).mkdir(parents=True, exist_ok=True)
    # Copy expected members individually; never extract paths from an archive.
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            path = Path(info.filename)
            if (
                path.suffix.lower() not in {".jpg", ".txt"}
                or path.parent.name != "train2017"
            ):
                continue
            folder = "images" if path.suffix.lower() == ".jpg" else "labels"
            if folder not in path.parts:
                continue
            with (
                zipped.open(info) as source,
                (dest / folder / path.name).open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
    (dest / "classes.txt").write_text("\n".join(names) + "\n")
    (dest / "provenance.txt").write_text(
        f"COCO128 real photographs and original COCO labels.\nSource: {COCO_URL}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if not (args.weights / "sam2.1_t.pt").is_file():
        download(VIDEO_URL, args.weights / "sam2.1_t.pt")
    if not (args.weights / "mobile_sam.pt").is_file():
        download(SAM_URL, args.weights / "mobile_sam.pt")
    if not (args.weights / "anchors/classes.txt").is_file():
        archive = args.archive or args.weights / "coco128.zip"
        if not archive.is_file():
            download(COCO_URL, archive)
        install_anchors(archive, args.weights / "anchors")
    print(
        f"MobileSAM, SAM 2.1 video tracking and real COCO128 rehearsal images ready in {args.weights.resolve()}"
    )


if __name__ == "__main__":
    main()
