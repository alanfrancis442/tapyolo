"""Frozen-backbone YOLO fine-tune with a 0.70 / 0.30 novel-anchor image mix."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from tapyolo.anchors import ANCHOR_COUNT
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
    plans = []
    for anchor_repeats in range(1, 33):
        center = max(1, round(ratio * n_anchor * anchor_repeats / n_novel))
        for novel_repeats in {max(1, center - 1), center, center + 1}:
            ns, ans = n_novel * novel_repeats, n_anchor * anchor_repeats
            fraction = ns / (ns + ans)
            plans.append(
                (abs(fraction - mu), ns + ans, novel_repeats, anchor_repeats, fraction)
            )
    close = [plan for plan in plans if plan[0] <= 0.015]
    best = min(close, key=lambda plan: (plan[1], plan[0])) if close else min(plans)
    return MixPlan(best[2], best[3], n_novel * best[2], n_anchor * best[3], best[4])


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
    replay_sources: list[tuple[Path, str]] | None = None,
) -> TrainResult:
    """Fine-tune a detection head. ``yolo_factory`` injects a fake for tests."""
    novel_root = Path(novel_dir)
    work = Path(project).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if anchor_dir is None:
        anchors = Path.cwd() / "weights" / "anchors"
        if not (anchors / "classes.txt").is_file():
            raise ValueError(
                "Real rehearsal images are missing. Run python -m tapyolo.setup, or pass --anchors with images/, labels/, classes.txt."
            )
    else:
        anchors = Path(anchor_dir)
    anchor_names = _anchor_names(anchors)
    names = list(anchor_names)
    for _, name in replay_sources or []:
        if not any(existing.casefold() == name.casefold() for existing in names):
            names.append(name)
    names, novel_class_id = _class_list(class_name, names)
    class_name = names[novel_class_id]
    canonical_replay = [
        (
            root,
            next(
                existing for existing in names if existing.casefold() == name.casefold()
            ),
        )
        for root, name in replay_sources or []
    ]
    dataset = work / "dataset"
    data_yaml, fraction = materialize_dataset(
        [(novel_root, class_name), *canonical_replay],
        anchors,
        dataset,
        names,
        mu,
        anchor_count=anchor_count,
    )
    params = TrainHyperParams(epochs=epochs, mu_novel=mu)
    overrides = build_train_overrides(
        params,
        data=str(data_yaml),
        device=device,
        project=str(work / "train"),
        imgsz=imgsz,
        batch=batch,
    )
    overrides["nbs"] = batch
    real_training = yolo_factory is None
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
    if real_training:
        from tapyolo.adaptation import adaptation_trainer

        yolo.train(trainer=adaptation_trainer(), **overrides)
    else:
        yolo.train(**overrides)
    save_dir = Path(
        getattr(getattr(yolo, "trainer", None), "save_dir", work / "train" / "head")
    )
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
        mix_fraction=fraction,
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
    data_yaml, _ = materialize_dataset(
        [(novel_root, class_name)], None, dataset, [class_name], 1.0
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
    save_dir = Path(
        getattr(getattr(yolo, "trainer", None), "save_dir", work / "train" / "head")
    )
    weights = _find_weights(save_dir)
    note = f"Trained {class_name} on the segmented object."
    return TrainResult(
        data_yaml=str(data_yaml),
        dataset_root=str(dataset),
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
    if not path.is_file():
        raise ValueError(
            "Rehearsal images need classes.txt so class ids can be aligned correctly"
        )
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names or len({name.casefold() for name in names}) != len(names):
        raise ValueError("classes.txt must contain distinct, non-empty class names")
    return names


def _class_list(class_name: str, anchor_names: list[str]) -> tuple[list[str], int]:
    names = list(anchor_names)
    for index, name in enumerate(names):
        if name.casefold() == class_name.casefold():
            return names, index
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


def materialize_dataset(sources, anchors, dataset, names, mu, *, anchor_count=25):
    """Keep every derivative of a source frame in one split; validate real originals only."""
    if dataset.exists():
        shutil.rmtree(dataset)
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    novel_train = []
    anchor_train = []
    seen = set()
    for source_index, (root, name) in enumerate(sources):
        root = Path(root).resolve()
        if (str(root), name) in seen:
            continue
        seen.add((str(root), name))
        pairs = _pairs(root)
        groups = {}
        for image, label in pairs:
            # Exported augmentations and object-removed negatives share their original's group.
            group = re.sub(r"_(?:bg\d+|gone)$", "", image.stem)
            groups.setdefault(group, []).append((image, label))
        positives = [
            g
            for g, items in groups.items()
            if any(lab.is_file() and lab.read_text().strip() for _, lab in items)
        ]
        if len(positives) < 2:
            raise ValueError(
                f"{name}: label at least two different keyframes for separate training and validation"
            )
        held = set(positives[::5])
        if len(held) == len(positives):
            held = {positives[-1]}
        negative_groups = [g for g in groups if g not in positives]
        held.update(negative_groups[::5])
        for group, items in groups.items():
            for image, label in items:
                stem = f"novel_{source_index}_{image.stem}"
                if group in held:
                    if image.stem != group:  # never validate a synthetic view
                        continue
                    _copy_pair(
                        image,
                        label,
                        dataset / "images/val",
                        dataset / "labels/val",
                        stem,
                        names.index(name),
                    )
                else:
                    novel_train.append((image, label, stem, names.index(name)))
    if anchors is not None:
        pairs = _pairs(Path(anchors))
        if len(pairs) < 2:
            raise ValueError("Rehearsal set needs at least two real images")
        # Fixed, non-overlapping subset. Unused originals remain held out.
        train_ids = set(
            np_even_indices(len(pairs), min(anchor_count, max(1, len(pairs) * 4 // 5)))
        )
        for index, (image, label) in enumerate(pairs):
            stem = f"anchor_{image.stem}"
            if index in train_ids:
                anchor_train.append((image, label, stem, None))
            else:
                _copy_pair(
                    image,
                    label,
                    dataset / "images/val",
                    dataset / "labels/val",
                    stem,
                    None,
                )
    plan = (
        plan_repetitions(len(novel_train), len(anchor_train), mu)
        if anchor_train
        else None
    )
    repeats = plan.novel_repeats if plan else 1
    for repeat in range(repeats):
        for image, label, stem, class_id in novel_train:
            _copy_pair(
                image,
                label,
                dataset / "images/train",
                dataset / "labels/train",
                f"{stem}_r{repeat}",
                class_id,
            )
    anchor_repeats = plan.anchor_repeats if plan else 0
    for repeat in range(anchor_repeats):
        for image, label, stem, class_id in anchor_train:
            _copy_pair(
                image,
                label,
                dataset / "images/train",
                dataset / "labels/train",
                f"{stem}_r{repeat}",
                class_id,
            )
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": names,
            },
            sort_keys=False,
        )
    )
    (dataset / "sources.json").write_text(
        json.dumps(
            [
                {"root": str(Path(root).resolve()), "name": name}
                for root, name in sources
            ],
            indent=2,
        )
    )
    count = len(novel_train) * repeats
    return data_yaml, count / (count + len(anchor_train) * anchor_repeats)


def np_even_indices(length, count):
    from tapyolo.sampler import evenly_spaced

    return evenly_spaced(list(range(length)), count)
