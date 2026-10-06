"""Frozen-backbone YOLO fine-tune with a 0.70 / 0.30 novel-anchor image mix."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from tapyolo.anchors import ANCHOR_COUNT, ANCHOR_NAMES, generate_anchor_set
from tapyolo.labels import remap_label_file

MU_NOVEL = 0.70
FREEZE_LAYERS = 10
LR0 = 0.001
DEFAULT_EPOCHS = 12
LABELED_EPOCHS = 40
LABELED_LR0 = 0.01
LOSS_MIX_NOTE = (
    "Novel and anchor images are mixed at μ=0.70 / 0.30 before the Ultralytics loader. "
    "The trainer applies one detection loss to that mixture. "
    "close_mosaic is 0 so mosaic probability 0.5 stays on through the short run. "
    "Optimizer is AdamW so Ultralytics does not replace lr0=0.001 with its auto fit."
)


class TrainHyperParams(BaseModel):
    """Product defaults match the 10–15 epoch head-tuning budget.

    ``epochs`` may be lower when a test calls the trainer with a tiny budget.
    """

    epochs: int = Field(default=DEFAULT_EPOCHS, ge=1, le=200)
    lr0: float = LR0
    freeze: int = FREEZE_LAYERS
    mosaic: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    scale: float = 0.5
    degrees: float = 10.0
    mu_novel: float = MU_NOVEL
    close_mosaic: int = 0


@dataclass(frozen=True)
class MixPlan:
    novel_repeats: int
    anchor_repeats: int
    novel_slots: int
    anchor_slots: int
    novel_fraction: float


@dataclass
class TrainResult:
    data_yaml: str
    dataset_root: str
    save_dir: str
    weights: str | None
    device: str
    epochs: int
    novel_class_id: int
    anchor_class_ids: list[int]
    mix_fraction: float
    note: str


def plan_repetitions(n_novel: int, n_anchor: int, mu: float = MU_NOVEL) -> MixPlan:
    """Repeat images so the novel fraction is near ``mu``."""
    if n_novel <= 0 or n_anchor <= 0:
        raise ValueError("novel and anchor sets must both be non-empty")
    if not 0.0 < mu < 1.0:
        raise ValueError("mu must be between 0 and 1")
    ratio = mu / (1.0 - mu)
    base = max(1, int(round(ratio * n_anchor / n_novel)))
    best: tuple[float, int, float, int, int] | None = None
    for repeats in range(max(1, base - 2), base + 3):
        novel_slots = n_novel * repeats
        anchor_slots = n_anchor
        fraction = novel_slots / (novel_slots + anchor_slots)
        error = abs(fraction - mu)
        if best is None or error < best[0]:
            best = (error, repeats, fraction, novel_slots, anchor_slots)
    assert best is not None
    return MixPlan(
        novel_repeats=best[1],
        anchor_repeats=1,
        novel_slots=best[3],
        anchor_slots=best[4],
        novel_fraction=best[2],
    )


def build_train_overrides(
    params: TrainHyperParams,
    *,
    data: str,
    device: str,
    project: str,
    imgsz: int,
    batch: int,
) -> dict:
    return {
        "data": data,
        "epochs": params.epochs,
        "lr0": params.lr0,
        "freeze": params.freeze,
        "mosaic": params.mosaic,
        "hsv_h": params.hsv_h,
        "hsv_s": params.hsv_s,
        "hsv_v": params.hsv_v,
        "scale": params.scale,
        "degrees": params.degrees,
        "close_mosaic": params.close_mosaic,
        "optimizer": "AdamW",
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "name": "head",
        "exist_ok": True,
        "workers": 0,
        "plots": False,
        "patience": params.epochs,
        "verbose": False,
        "amp": device.startswith("cuda"),
    }


def train_detection_head(
    novel_dir: Path,
    class_name: str,
    *,
    epochs: int = DEFAULT_EPOCHS,
    model: str = "yolo11n.pt",
    device: str = "cpu",
    project: Path | str = "runs/tapyolo",
    imgsz: int = 640,
    batch: int = 8,
    anchor_dir: Path | None = None,
    anchor_count: int = ANCHOR_COUNT,
    mu: float = MU_NOVEL,
    yolo_factory=None,
    progress_callback=None,
) -> TrainResult:
    """Fine-tune a detection head. ``yolo_factory`` injects a fake for tests."""
    novel_root = Path(novel_dir)
    work = Path(project).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if anchor_dir is None:
        anchors = generate_anchor_set(work / "anchors", count=anchor_count)
    else:
        anchors = Path(anchor_dir)
    anchor_names = _anchor_names(anchors)
    names, novel_class_id = _class_list(class_name, anchor_names)
    plan = plan_repetitions(_count_images(novel_root), _count_images(anchors), mu)
    dataset = work / "dataset"
    data_yaml = _materialize_mix(novel_root, anchors, dataset, names, novel_class_id, plan)
    params = TrainHyperParams(epochs=epochs, mu_novel=mu)
    overrides = build_train_overrides(
        params,
        data=str(data_yaml),
        device=device,
        project=str(work / "train"),
        imgsz=imgsz,
        batch=batch,
    )
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    yolo = yolo_factory(model)
    state = {"started": None, "done": 0}

    def on_train_epoch_end(trainer) -> None:
        import time

        if state["started"] is None:
            state["started"] = time.perf_counter()
        state["done"] += 1
        total = int(getattr(trainer, "epochs", epochs))
        elapsed = time.perf_counter() - state["started"]
        remaining = max(0.0, (total - state["done"]) * (elapsed / state["done"]))
        if progress_callback is not None:
            progress_callback(state["done"], total, remaining)

    if hasattr(yolo, "add_callback"):
        yolo.add_callback("on_train_epoch_end", on_train_epoch_end)
    yolo.train(**overrides)
    save_dir = Path(getattr(getattr(yolo, "trainer", None), "save_dir", work / "train" / "head"))
    weights = _find_weights(save_dir)
    return TrainResult(
        data_yaml=str(data_yaml),
        dataset_root=str(dataset),
        save_dir=str(save_dir),
        weights=str(weights) if weights else None,
        device=device,
        epochs=epochs,
        novel_class_id=novel_class_id,
        anchor_class_ids=list(range(len(anchor_names))),
        mix_fraction=plan.novel_fraction,
        note=LOSS_MIX_NOTE,
    )


def train_labeled_class(
    novel_dir: Path,
    class_name: str,
    *,
    epochs: int = LABELED_EPOCHS,
    model: str = "yolo11n.pt",
    device: str = "cpu",
    project: Path | str = "runs/tapyolo",
    imgsz: int = 640,
    batch: int = 4,
    yolo_factory=None,
    progress_callback=None,
) -> TrainResult:
    """Train one new class from the boxes the user drew.

    Ultralytics scales ``lr0`` by ``batch / nbs`` and prefixes a relative project
    with its global runs directory. This run keeps the requested learning rate
    and writes weights under ``project``. Mosaic stays off because a handful of
    orientations is not enough to survive a four-image mosaic.
    """
    novel_root = Path(novel_dir).resolve()
    work = Path(project).resolve()
    dataset = work / "labeled-data"
    dataset.mkdir(parents=True, exist_ok=True)
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(novel_root),
                "train": "images",
                "val": "images",
                "names": {0: class_name},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    safe_batch = max(1, int(batch))
    overrides = {
        "data": str(data_yaml),
        "epochs": epochs,
        "lr0": LABELED_LR0,
        "lrf": 0.1,
        "freeze": FREEZE_LAYERS,
        "mosaic": 0.0,
        "mixup": 0.0,
        "close_mosaic": 0,
        "optimizer": "AdamW",
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "scale": 0.5,
        "degrees": 15.0,
        "imgsz": imgsz,
        "batch": safe_batch,
        "nbs": safe_batch,
        "warmup_epochs": 1,
        "device": device,
        "project": str(work / "train"),
        "name": "head",
        "exist_ok": True,
        "workers": 0,
        "plots": False,
        "patience": epochs,
        "verbose": False,
        "amp": str(device).startswith("cuda"),
    }
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    yolo = yolo_factory(model)
    state = {"started": None, "done": 0}

    def on_train_epoch_end(trainer) -> None:
        import time

        if state["started"] is None:
            state["started"] = time.perf_counter()
        state["done"] += 1
        total = int(getattr(trainer, "epochs", epochs))
        elapsed = time.perf_counter() - state["started"]
        remaining = max(0.0, (total - state["done"]) * (elapsed / state["done"]))
        if progress_callback is not None:
            progress_callback(state["done"], total, remaining)

    if hasattr(yolo, "add_callback"):
        yolo.add_callback("on_train_epoch_end", on_train_epoch_end)
    yolo.train(**overrides)
    save_dir = Path(getattr(getattr(yolo, "trainer", None), "save_dir", work / "train" / "head"))
    weights = _find_weights(save_dir)
    note = f"Trained {class_name} on the segmented object."
    return TrainResult(
        data_yaml=str(data_yaml),
        dataset_root=str(novel_root),
        save_dir=str(save_dir),
        weights=str(weights) if weights else None,
        device=device,
        epochs=epochs,
        novel_class_id=0,
        anchor_class_ids=[],
        mix_fraction=1.0,
        note=note,
    )


def _anchor_names(anchor_root: Path) -> list[str]:
    path = anchor_root / "classes.txt"
    if path.is_file():
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    return list(ANCHOR_NAMES)


def _class_list(class_name: str, anchor_names: list[str]) -> tuple[list[str], int]:
    names = list(anchor_names)
    if class_name in names:
        return names, names.index(class_name)
    names.append(class_name)
    return names, len(names) - 1


def _count_images(root: Path) -> int:
    return len(_pairs(root))


def _pairs(root: Path) -> list[tuple[Path, Path]]:
    image_dir = root / "images"
    if not image_dir.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        pairs.append((image, root / "labels" / f"{image.stem}.txt"))
    return pairs


def _materialize_mix(
    novel_root: Path,
    anchor_root: Path,
    dataset: Path,
    names: list[str],
    novel_class_id: int,
    plan: MixPlan,
) -> Path:
    if dataset.exists():
        shutil.rmtree(dataset)
    image_dir = dataset / "images" / "train"
    label_dir = dataset / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    novel_pairs = _pairs(novel_root)
    anchor_pairs = _pairs(anchor_root)
    for repeat in range(plan.novel_repeats):
        for image, label in novel_pairs:
            stem = f"novel_{image.stem}_r{repeat}"
            _copy_pair(image, label, image_dir, label_dir, stem, novel_class_id)
    for repeat in range(plan.anchor_repeats):
        for image, label in anchor_pairs:
            stem = f"anchor_{image.stem}_r{repeat}"
            _copy_pair(image, label, image_dir, label_dir, stem, None)
    data_yaml = dataset / "data.yaml"
    payload = {
        "path": str(dataset),
        "train": "images/train",
        "val": "images/train",
        "names": names,
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return data_yaml


def _copy_pair(
    image: Path,
    label: Path,
    image_dir: Path,
    label_dir: Path,
    stem: str,
    remap_to: int | None,
) -> None:
    shutil.copyfile(image, image_dir / f"{stem}{image.suffix.lower()}")
    dest = label_dir / f"{stem}.txt"
    if remap_to is None:
        if label.is_file():
            shutil.copyfile(label, dest)
        else:
            dest.write_bytes(b"")
    else:
        remap_label_file(label, dest, remap_to)


def _find_weights(save_dir: Path) -> Path | None:
    for relative in (Path("weights/best.pt"), Path("weights/last.pt")):
        candidate = save_dir / relative
        if candidate.is_file():
            return candidate
    if save_dir.is_dir():
        found = sorted(save_dir.rglob("best.pt"))
        if found:
            return found[-1]
    return None
