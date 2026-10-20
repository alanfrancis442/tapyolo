"""Expand a YOLO detection head without rebuilding its learned hidden layers."""

from __future__ import annotations

import copy
import math


def expand_detector(source, names):
    import torch
    from torch import nn

    model = copy.deepcopy(source).float()
    names = dict(enumerate(names)) if isinstance(names, list) else dict(names)
    source_names = {
        str(name).strip().casefold(): int(index) for index, name in source.names.items()
    }
    head = model.model[-1]
    if not hasattr(head, "cv3") or getattr(head, "end2end", False):
        raise ValueError(
            "Head adaptation currently supports standard YOLO11/YOLOv8 detection models"
        )
    with torch.no_grad():
        for scale, branch in enumerate(head.cv3):
            old = branch[-1]
            if not isinstance(old, nn.Conv2d) or old.kernel_size != (1, 1):
                raise ValueError(
                    "Unsupported classification head; use YOLO11n or YOLOv8n"
                )
            expanded = nn.Conv2d(old.in_channels, len(names), 1, bias=True).to(
                old.weight.device
            )
            nn.init.normal_(expanded.weight, std=0.01)
            expanded.bias.fill_(
                math.log(5 / len(names) / (640 / float(head.stride[scale])) ** 2)
            )
            for destination, name in names.items():
                original = source_names.get(str(name).strip().casefold())
                if original is not None:
                    expanded.weight[destination].copy_(old.weight[original])
                    expanded.bias[destination].copy_(old.bias[original])
            branch[-1] = expanded
    head.nc = len(names)
    head.no = head.reg_max * 4 + len(names)
    model.names = names
    model.yaml["nc"] = len(names)
    if hasattr(model, "criterion"):
        del model.criterion
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def adaptation_trainer():
    """Use the supported trainer hook while retaining the checkpoint architecture."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class AdaptationTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is None:
                return super().get_model(cfg, weights, verbose)
            return expand_detector(weights, self.data["names"])

    return AdaptationTrainer
