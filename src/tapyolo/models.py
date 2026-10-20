"""A confined catalog of trained checkpoints; source datasets survive deletion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def catalog(root: Path, base: str) -> list[dict]:
    resolved = root.resolve()
    records = []
    for path in root.rglob("best.pt"):
        actual = path.resolve()
        if (
            not path.is_file()
            or path.parent.name != "weights"
            or not actual.is_relative_to(resolved)
            or actual == Path(base).resolve()
        ):
            continue
        # Refuse symlinked paths, including symlinked run folders.
        if any(
            part.is_symlink() for part in [path, *path.parents] if part != root.parent
        ):
            continue
        relative = actual.relative_to(resolved).as_posix()
        records.append(
            {
                "id": hashlib.sha256(relative.encode()).hexdigest()[:20],
                "path": str(path),
                "name": relative,
                "modified": path.stat().st_mtime,
            }
        )
    return sorted(records, key=lambda item: item["modified"], reverse=True)


def replay_sources(weights: str, root: Path) -> list[tuple[Path, str]]:
    """Read our provenance or recover a legacy single-class UI dataset."""
    checkpoint = Path(weights).resolve()
    if not checkpoint.is_relative_to(root.resolve()):
        return []
    metadata = checkpoint.parent / "tapyolo.json"
    if metadata.is_file():
        data = json.loads(metadata.read_text())
        sources = data.get("sources", [])
        if not sources:
            raise ValueError(
                "This model has no saved replay sources. Select the base model before training."
            )
    else:
        args = checkpoint.parent.parent / "args.yaml"
        if not args.is_file():
            raise ValueError(
                "This model has no replay provenance. Select the base model to start a new collection of classes."
            )
        data_path = Path(yaml.safe_load(args.read_text()).get("data", ""))
        if not data_path.is_file():
            raise ValueError(
                "The previous model's dataset is missing. Select another model before training."
            )
        data = yaml.safe_load(data_path.read_text())
        names = data.get("names", {})
        if len(names) != 1:
            raise ValueError(
                "This legacy multi-class model has no replay provenance. Select the base model or a newer trained model."
            )
        name = names.get(0, names.get("0")) if isinstance(names, dict) else names[0]
        sources = [{"root": data["path"], "name": name}]
    found = []
    for item in sources:
        path = Path(item["root"]).resolve()
        if not path.is_relative_to(root.resolve()) or not (path / "images").is_dir():
            raise ValueError(
                "A previous class's replay images are missing; select another model before training"
            )
        found.append((path, item["name"]))
    return found
